import matplotlib
import torch
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn import preprocessing

from typing import List, Generator, NoReturn

from . import EmbeddedFeatures

class Analyzer:
  """ A help class to provide methods for embeddings analysis
  """

  def __init__(self):
    sns.set(style = 'white')
    pass



  def pca(self, embedded: EmbeddedFeatures) -> Generator[int, npt.ArrayLike, NoReturn]:
    """Compute PCA decomposition

    Args:
        embedded (EmbeddedFeatures): A 1-block embedding features

    Yields:
        Generator[int, npt.ArrayLike, NoReturn]: PCA for each block
    """
    for index in range(embedded.token_embeddings.shape[0]):
      attention_mask = embedded.attention_mask[index]

      embedding = embedded.token_embeddings[index][attention_mask == 1]

      yield PCA(random_state=0).fit_transform(embedding)

  def cosine_sim_sentence(self, embedded: EmbeddedFeatures, layer: int) -> Generator[int, npt.ArrayLike, NoReturn]:
    """ Calculates a cosine similarity between each pair of words in the embedding.
        Then plots a heatmap of cosine similarity

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

      similarity = cosine_similarity(embedding.numpy(), dense_output = False)

      yield similarity

  def cosine_sim(self, lhs: torch.Tensor, rhs: torch.Tensor):
    """ Calculates a standard cosine similarity of vectors

    Args:
        lhs (EmbeddedFeatures): An embedding after pass to transformer
        rhs (EmbeddedFeatures): An embedding after pass to transformer
    """

    return torch.nn.functional.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0))

  def plot_heatmap(self, matrix: npt.ArrayLike, xticklabels: List[str], yticklabels: List[str]):
    """ Plot matrix heatmap

    """

    figsize = (10, 10)

    if len(matrix.shape) == 1:
      figsize = (int(matrix.shape[0]), 3)
      matrix = matrix.reshape(1, matrix.shape[0])
    else:
      figsize = (int(matrix.shape[0]), int(matrix.shape[1])

    figure = plt.figure(figsize = figsize)
    axis = plt.subplot(111)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    heatmap = sns.heatmap(matrix,
                          cmap = cmap,
                          annot = True,
                          fmt = '.2f',
                          linewidth = .5,
                          cbar_kws={"shrink": .5},
                          square = True,
                          ax = axis)

    heatmap.set_xticklabels(xticklabels, rotation = 90, fontsize = 20)
    heatmap.set_yticklabels(yticklabels, rotation = 0, fontsize = 20)

    plt.tight_layout()

    axis.set_xlabel("")
    axis.set_ylabel("")

    return axis

  def plot_scatter(self, matrix: npt.ArrayLike, labels: List[str], normalize = False, axis = None):
    """ Plots scatter
    """

    if normalize:
      matrix = preprocessing.normalize(matrix)

    new_axis = None

    if axis is None:
      figsize = (20, 20)
      figure = plt.figure(figsize = figsize)

      new_axis = plt.subplot(111)
    else:
      new_axis = axis

    sns.scatterplot(x = matrix[:, 0], y = matrix[:, 1], size = 1000, ax = new_axis, legend = False)

    for i in range(len(labels)):
      plt.annotate(
        labels[i],
        xy = (matrix[i][0], matrix[i][1]),
        xytext = (5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom',
        fontsize = 12
      )

    plt.tight_layout()

    return new_axis
