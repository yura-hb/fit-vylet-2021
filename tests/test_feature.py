
import pytest
import os
import torch

from codevec.models import Transformer
from codevec.utils import EmbeddedFeatures, RawFeatures

class TestFeature:

  output_dir: str = 'test_output'
  input_features = output_dir + '/input_features.pt'
  output_embeddings = output_dir + '/output_embeddings.pt'

  def setup(self):
    if not os.path.exists(self.output_dir):
      os.mkdir(self.output_dir)

    if not os.path.exists(self.input_features):
      auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
      action_config = Transformer.ActionConfig()
      split_config = Transformer.SplitConfig(128)

      bert_model = Transformer.auto_model(auto_config, action_config, split_config)

      text = """
         Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in 
         the evening
         we were freezed with minus 30 and strong snow. Everything was ok.
         """
      texts = [text * 100, text * 20, text]

      features = bert_model.tokenize(texts)
      embedding = bert_model(features)

      features.write(self.input_features)
      embedding.write(self.output_embeddings)

  def test_loading(self):
    raw = RawFeatures.read(self.input_features)
    embedded = EmbeddedFeatures.read(self.output_embeddings)

    print(embedded.token_embeddings.shape)

    assert len(torch.unique(raw.sample_mapping)) == 3, 'Only 3 samples are presented in the features'
    assert torch.all(raw.attention_mask == embedded.attention_mask), 'Attention mask must be the same'
    assert torch.all(raw.sample_mapping == embedded.sample_mapping), 'Sample mapping must be the same'

  def test_raw_at_operator(self):
    raw = RawFeatures.read(self.input_features)
    embedded = EmbeddedFeatures.read(self.output_embeddings)

    assert torch.all(raw.at([0]).input_ids == raw.input_ids[0]), 'Elements must be equally selected'
    assert raw.at([0]).input_ids.shape != raw.input_ids[0].shape, 'But their shapes are unequal'

    at = raw.at([0])
    at.input_ids[0][0] = -1

    assert raw.input_ids[0][0] != at.input_ids[0][0], 'They aren\'t copies'

  def test_embedded_at_operator(self):
    embedded = EmbeddedFeatures.read(self.output_embeddings)

    assert torch.all(embedded.at([0]).token_embeddings == embedded.token_embeddings[0]), 'Elements must be equally selected'
    assert embedded.at([0]).token_embeddings.shape != embedded.token_embeddings[0].shape, 'But their shapes are unequal'

    at = embedded.at([0])
    at.token_embeddings[0][0][0] = -1

    assert embedded.token_embeddings[0][0][0] != at.token_embeddings[0][0][0], 'They aren\'t copies'

  def test_embedded_read_write(self):
    raw = RawFeatures.read(self.input_features)
    embedded = EmbeddedFeatures.read(self.output_embeddings)

    assert raw.input_ids.shape == embedded.token_embeddings.shape[:2]
    assert embedded.token_embeddings.shape[:2] == embedded.attention_mask.shape

  def test_embedded_write_flattening(self):
    d = torch.load(self.output_embeddings)

    assert 'token_embeddings' in d.keys()
    assert 'embedding_shape' in d.keys()
    assert 'attention_mask' in d.keys()

    embeddings = d['token_embeddings']

    assert embeddings.shape[0] > 512, "An flatten embeddings shape must be bigger than block size"
    assert embeddings.shape[1] == 768, "An embedding vector must be 768 length"

