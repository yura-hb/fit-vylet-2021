import pytest
from codevec.models.Transformer import *

class TestTransformer:

  def test_bert_model_encoding(self):
    text = """
    Yesterday was a beautiful day. In the morning it was 30 degrees hot. In the afternoon it was raining and in the evening
    we were freezed with minus 30 and strong snow. Everything was ok.
    """

    config = Transformer.Config('bert-base-cased', 'bert-base-cased', model_args = { 'output_hidden_states': True })
    bert_model = Transformer(config)

    tokenized = bert_model.tokenize([text])
    embedding = bert_model(tokenized)

    print(embedding)

    assert isinstance(embedding, dict)
    assert 'token_embeddings' in embedding.keys()
    assert 'hidden_states' in embedding.keys()
    assert 'cls_token' in embedding.keys()
    assert len(embedding['token_embeddings']) == len(tokenized['input_ids'])



