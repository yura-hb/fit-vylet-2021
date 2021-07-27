import torch
import numpy.typing as npt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from typing import Generator

from . import EmbeddedFeatures


class SimilarityMethod:
  """ A help class to provide methods for embeddings analysis
  """

  def __init__(self):
    pass

  @staticmethod
  def pca(embedded: EmbeddedFeatures, layer: int) -> Generator[int, npt.ArrayLike, None]:
    """Compute PCA decomposition

    Args:
        embedded (EmbeddedFeatures): A 1-block embedding features
        normalize (bool): If data should be normalized

    Yields:
        Generator[int, npt.ArrayLike, NoReturn]: PCA for each block
    """

    embeddings = embedded.token_embeddings if layer is None else embedded.hidden_states[layer]

    for index in range(embeddings.shape[0]):
      attention_mask = embedded.attention_mask[index]

      embedding = embeddings[index][attention_mask == 1]

      pca = PCA(random_state=0).fit_transform(embedding)

      yield pca

  @staticmethod
  def cosine_sim_sentence(embedded: EmbeddedFeatures, layer: int) -> Generator[int, npt.ArrayLike, None]:
    """ Calculates a cosine similarity between each pair of words in the embedding.

    Args:
        embedded (EmbeddedFeatures): A 1-block embedding features
        layer: int: A layer to take data from. In case, if None, the method will take the last layer

    Yields:
        Generator[int, npt.ArrayLike, NoReturn]: PCA for each block
    """

    embeddings = embedded.token_embeddings if layer is None else embedded.hidden_states[layer]

    for index in range(embeddings.shape[0]):
      attention_mask = embedded.attention_mask[index]

      embedding = embeddings[index][attention_mask == 1]

      similarity = cosine_similarity(embedding.numpy(), dense_output=False)

      yield similarity

  @staticmethod
  def cosine_sim(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """ Calculates a standard cosine similarity of vectors

    Args:
        lhs (EmbeddedFeatures): An embedding after pass to transformer
        rhs (EmbeddedFeatures): An embedding after pass to transformer
    """

    return torch.nn.functional.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0))
