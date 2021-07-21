import pytest

from codevec.models.Transformer import *
from codevec.utils.Features import *

import torch

class TestTransformer:

  def test_bert_model_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    config = Transformer.Config('bert-base-cased', 'bert-base-cased', True, model_args = { 'output_hidden_states': False })
    bert_model = Transformer(config)

    features = bert_model.tokenize([text])
    embedding = bert_model(features)

    assert isinstance(features, RawFeatures)
    assert isinstance(embedding, EmbeddedFeatures)
    assert len(embedding.token_embeddings) == len(features.input_ids)

  def test_bert_device_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    config = Transformer.Config('bert-base-cased', 'bert-base-cased', True, model_args = { 'output_hidden_states': False })
    bert_model = Transformer(config)

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

    config = Transformer.Config('bert-base-cased', 'bert-base-cased', True, model_args = { 'output_hidden_states': False }, autograd = True)
    bert_model = Transformer(config)
    features = bert_model.tokenize([text])
    embedding = bert_model(features)

    assert embedding.token_embeddings.requires_grad, "Grad must work"

    config = Transformer.Config('bert-base-cased', 'bert-base-cased', True, model_args = { 'output_hidden_states': False }, autograd = False)
    bert_model = Transformer(config)
    features = bert_model.tokenize([text])
    embedding = bert_model(features)

    assert not embedding.token_embeddings.requires_grad, "Grad shouldn't work"
