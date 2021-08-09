import pytest

from codevec.models.Transformer import *
from codevec.models.BSSM import *

class TestBSSM:

  def test_bssm_encoding(self):
    text_a = "Some long day will end soon"
    text_b = "The day must to end soon"

    auto_config = Transformer.AutoConfig('bert-base-cased', 'bert-base-cased')
    action_config = Transformer.ActionConfig(output_hidden_states=True, autograd=False)
    split_config = Transformer.SplitConfig(128)

    bert_model = Transformer.auto_model(auto_config, action_config, split_config)

    tokenized = bert_model.tokenize([text_a, text_b])
    tokenized_len = tokenized.input_ids.shape[1]

    embedding = bert_model(tokenized)

    bssm = BSSM()
    vector = bssm(embedding)

    assert vector.shape[0] == 9 * tokenized_len
