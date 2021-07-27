import dash

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from jupyter_dash import JupyterDash

import plotly.graph_objects as go

from codevec.models.Transformer import Transformer
from codevec.utils.SimilarityMethod import SimilarityMethod
from codevec.utils.Plotter import Plotter

from sklearn import preprocessing

import numpy as np


class Workspace:

  _heatmap_plot = 'heatmap'
  _pca_scatter_plot = 'pca'
  _pca_scatter_3d_plot = 'pca_3d'

  def __init__(self, model_path: str):
    config = Transformer.Config(model_path,
                                model_path,
                                True,
                                model_args={'output_hidden_states': True},
                                autograd=False)

    self.transformer = Transformer(config)

  def jupyter_run(self, text: str):
    self.app = JupyterDash(__name__,
                           external_stylesheets=[dbc.themes.BOOTSTRAP],
                           suppress_callback_exceptions=True)

    self.__build(text)

    self.app.run_server(mode='inline', debug=True)

  def webserver_run(self, text: str):
    self.app = dash.Dash(__name__,
                         external_stylesheets=[dbc.themes.BOOTSTRAP],
                         suppress_callback_exceptions=True)

    self.__build(text)

    self.app.run_server(debug=True)

  def __build(self, text: str):
    data = self.__process_text(text)

    button_options = []

    for i in range(data[0].shape[0]):
      button_options.append({'label': 'Layer {}'.format(i), 'value': i})

    method_options = [
      {'label': 'Cosine similarity', 'value': self._heatmap_plot},
      {'label': 'PCA 2d', 'value': self._pca_scatter_plot},
      {'label': 'PCA 3d', 'value': self._pca_scatter_3d_plot},
    ]

    #
    # Layout
    #

    self.app.layout = html.Div([
      html.Div([
        html.Div([
          dcc.Dropdown(id='layer-dropdown',
                       options=button_options,
                       value=0)
        ], className='layer-dropdown'),
        html.Div([
          dcc.Dropdown(id='method-dropdown',
                       options=method_options,
                       value=self._heatmap_plot)
        ], className='method-dropdown'),
        html.H3(id='some-debug-text')
      ], className='dropdown-container'),

      html.Div([
        dcc.Graph(id='graph', responsive=True, className='graph')
      ], className='graph-container'),

      dcc.Store(id='report', data=data)
    ], className='app-container')

    #
    # Graph
    #

    @ self.app.callback(Output('some-debug-text', 'children'),
                        Output('graph', 'figure'),
                        [Input('layer-dropdown', 'value'), Input('method-dropdown', 'value')],
                        State('report', 'data'))
    def select_layer(layer, method, report):
      return '{} {}'.format(1, 2), self.make_figure(layer, method, report)

  def make_figure(self, layer, method, report):
    similarity, normalized_2d, normalized_3d, indexed_tokens = report

    figure = go.Figure()

    figure.update_layout(template='ggplot2',
                         xaxis=dict(showgrid=False),
                         yaxis=dict(showgrid=False))

    if method == self._heatmap_plot:
      figure.add_trace(Plotter.heatmap(similarity[layer], indexed_tokens, indexed_tokens))
    elif method == self._pca_scatter_plot:
      figure.add_trace(Plotter.scatter(np.array(normalized_2d[layer]), indexed_tokens))
      figure.update_layout(
        xaxis=dict(range=[-2, 2]),
        yaxis=dict(range=[-2, 2])
      )
    elif method == self._pca_scatter_3d_plot:
      figure.add_trace(Plotter.scatter_3d(np.array(normalized_3d[layer]), indexed_tokens))

      figure.update_layout(
        scene=dict(xaxis=dict(range=[-2, 2]),
                   yaxis=dict(range=[-2, 2]),
                   zaxis=dict(range=[-2, 2]))
      )
    else:
      pass

    return figure

  def __process_text(self, text: str):
    features = self.transformer.tokenize(text)
    embedding = self.transformer(features)

    tokens = self.transformer.tokenizer.convert_ids_to_tokens(features.input_ids[0])

    indexed_tokens = ['{}_{}'.format(id, token)
                      for id, token in enumerate(tokens)]

    similarity = np.array([next(SimilarityMethod.cosine_sim_sentence(embedding, index))
                           for index in range(len(embedding.hidden_states))])
    pca = np.array([next(SimilarityMethod.pca(embedding, index))
                    for index in range(len(embedding.hidden_states))])

    normalized_2d = np.array(list(map(lambda matrix: preprocessing.normalize(matrix[:, :2]), pca)))
    normalized_3d = np.array(list(map(lambda matrix: preprocessing.normalize(matrix[:, :3]), pca)))

    return similarity, normalized_2d, normalized_3d, indexed_tokens


if __name__ == '__main__':
  workspace = Workspace('bert-base-cased')

  workspace.webserver_run('Some text lovely large text')
