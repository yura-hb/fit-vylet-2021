import pytest
from codevec.models.Transformer import *
from codevec.utils.Features import *

class TestTransformer:

  def test_bert_model_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    config = Transformer.Config('bert-base-cased', 'bert-base-cased', True, model_args = { 'output_hidden_states': True })
    bert_model = Transformer(config)

    features = bert_model.tokenize([text])
    embedding = bert_model(features)

    assert isinstance(embedding, Features)
    assert len(embedding.token_embeddings) == len(embedding.input_ids)
