import plotly.graph_objects as go
import numpy.typing as npt

from typing import List


class Plotter:

  def __init__(self):
    pass

  @staticmethod
  def heatmap(matrix: npt.ArrayLike, xtickslabels: List, ytickslabels: List):
    return go.Heatmap(z=matrix,
                      x=xtickslabels,
                      y=ytickslabels,
                      name="Heatmap")

  @staticmethod
  def scatter(matrix: npt.ArrayLike, labels: List = []):
    return go.Scatter(x=matrix[:, 0],
                      y=matrix[:, 1],
                      mode='markers+text',
                      text=labels,
                      textposition="top center")

  @staticmethod
  def scatter_3d(matrix: npt.ArrayLike, labels: List = []):
    return go.Scatter3d(x=matrix[:, 0],
                        y=matrix[:, 1],
                        z=matrix[:, 2],
                        mode='markers+text',
                        text=labels,
                        textposition="top center")
