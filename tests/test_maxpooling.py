
from codevec.models.MaxPooling import MaxPooling
from codevec.models.Transformer import Transformer

class TestMaxPooling:

  def test_bert_max_pooling(self):
    text_a = "Some long day will end soon"

    config = Transformer.Config('bert-base-cased', 'bert-base-cased', model_args = { 'output_hidden_states': True })
    bert_model = Transformer(config)

    tokenized = bert_model.tokenize([text_a])
    embedding = bert_model(tokenized)

    column_pooling = MaxPooling(1)
    row_pooling = MaxPooling(2)

    copy = embedding.token_embeddings.clone()
    embedding_length = copy.shape[2]

    assert row_pooling(embedding).token_embeddings.shape == tokenized.input_ids.shape

    embedding.token_embeddings = copy

    assert column_pooling(embedding).token_embeddings.shape[1] == embedding_length
