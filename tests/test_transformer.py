import pytest

from codevec.models.Transformer import *
from codevec.utils.RawFeatures import *

import torch

class TestTransformer:

  def test_bert_model_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
    action_config = Transformer.ActionConfig(output_hidden_states=True)
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    features = bert_model.tokenize(text)
    embedding = bert_model(features)

    assert isinstance(features, RawFeatures)
    assert isinstance(embedding, EmbeddedFeatures)
    assert embedding.hidden_states.numel() > 0, "Hidden states must be set"
    assert len(embedding.token_embeddings) == len(features.input_ids)

  def test_roberta_model_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    auto_config = Transformer.AutoConfig('roberta-base', 'roberta-base')
    action_config = Transformer.ActionConfig()
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    features = bert_model.tokenize(text)
    embedding = bert_model(features)

    assert isinstance(features, RawFeatures)
    assert isinstance(embedding, EmbeddedFeatures)
    assert len(embedding.token_embeddings) == len(features.input_ids)

  def test_gpt2_model_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    auto_config = Transformer.AutoConfig('gpt2', 'gpt2')
    action_config = Transformer.ActionConfig(requires_additional_pad_token=True)
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    features = bert_model.tokenize(text)
    embedding = bert_model(features)

    assert isinstance(features, RawFeatures)
    assert isinstance(embedding, EmbeddedFeatures)
    assert len(embedding.token_embeddings) == len(features.input_ids)

  def test_bert_multiple_file_encoding(self):
    text = """
        Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in 
        the evening
        we were freezed with minus 30 and strong snow. Everything was ok.
        """

    texts = [text * 20, text * 20, text]

    auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
    action_config = Transformer.ActionConfig()
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    features = bert_model.tokenize(texts)
    embedding = bert_model(features)

    assert isinstance(features, RawFeatures)
    assert isinstance(embedding, EmbeddedFeatures)
    assert len(embedding.token_embeddings) == len(features.input_ids)
    assert features.sample_mapping.numel() != 0

  def test_bert_device_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
    action_config = Transformer.ActionConfig()
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    features = bert_model.tokenize([text])
    features = features.to(device)

    assert features.input_ids.device == device
    assert features.attention_mask.device == device
    assert features.token_type_ids.device == device

    bert_model = bert_model.to(device)

    embedding = bert_model(features)

    assert isinstance(features, RawFeatures)
    assert isinstance(embedding, EmbeddedFeatures)
    assert len(embedding.token_embeddings) == len(features.input_ids)

  def test_no_grad_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
    action_config = Transformer.ActionConfig(autograd=True)
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    features = bert_model.tokenize([text])
    embedding = bert_model(features)

    assert embedding.token_embeddings.requires_grad, "Grad must work"

    auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
    action_config = Transformer.ActionConfig(autograd=False)
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    features = bert_model.tokenize([text])
    embedding = bert_model(features)

    assert not embedding.token_embeddings.requires_grad, "Grad shouldn't work"

  def test_overflow_tokenization(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """ * 100

    split_config = None
    features = self.get_bert_features(text, None)

    assert features.input_ids.shape[0] == 1

    split_config = Transformer.SplitConfig(0)
    features = self.get_bert_features(text, split_config)

    assert features.input_ids.shape[0] == 9

    split_config = Transformer.SplitConfig(128)
    features = self.get_bert_features(text, split_config)

    assert features.input_ids.shape[0] == 11

  def get_bert_features(self, text: str, split_config: Transformer.SplitConfig):
    auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
    action_config = Transformer.ActionConfig(autograd=False)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    return bert_model.tokenize([text])
