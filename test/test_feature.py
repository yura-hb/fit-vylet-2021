from codevec.utils.Features import EmbeddedFeatures

import pytest
import os
import torch

from codevec.utils.Features import EmbeddedFeatures

class TestFeature:

  def test_load_save(self):
    features = EmbeddedFeatures(
      token_embeddings = torch.tensor([1, 2, 3]),
      attention_mask = torch.tensor([1, 1, 1]),
      cls_token = torch.tensor([0, 1, 1]),
      hidden_states = torch.tensor([0, 0, 0])
    )

    path = 'test.pt'

    features.write(path)

    loaded_features = EmbeddedFeatures.read(path)

    assert torch.all(features.token_embeddings == loaded_features.token_embeddings)
    assert torch.all(features.attention_mask == loaded_features.attention_mask)
    assert torch.all(features.cls_token == loaded_features.cls_token)
    assert torch.all(features.hidden_states == loaded_features.hidden_states)

    os.remove(path)
