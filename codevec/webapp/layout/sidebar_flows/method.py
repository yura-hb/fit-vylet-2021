import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dash.dependencies import Input, Output, State

from utils.constants import ElementId, SimilarityMethodKey
from app import app, Context

from dash.exceptions import PreventUpdate

from codevec.utils import SimilarityMethod, Plotter

import numpy as np

from sklearn import preprocessing

layout = html.Div([
  html.H2("3. Select method",
          id=ElementId.method_label.value,
          className=ElementId.method_label.value),

  html.Div([], className=ElementId.method_separator.value),

  html.Div([
    dcc.Dropdown(id=ElementId.method_layer_dropdown.value,
                 options=[])
  ], className=ElementId.method_layer_dropdown.value),

  html.Div([
    dcc.Dropdown(id=ElementId.method_dropdown.value,
                 options=[])
  ], className=ElementId.method_dropdown.value),

  html.Div([
    dbc.Button("Visualize", id=ElementId.method_confirm_button.value, color="primary", className="mr-1", block=True)
  ], className=ElementId.method_confirm_button.value),

  html.Div([
    dbc.Button("Finish", color="danger", className="mr-1", block=True, href=ElementId.model_input_path.value)
  ], className=ElementId.method_finish_button.value),

  html.Div(id='hidden', style={'display': 'none'})
])


@app.callback(Output(ElementId.method_layer_dropdown.value, 'options'),
              Output(ElementId.method_dropdown.value, 'options'),
              Input(ElementId.is_embedding_ready.value, 'data'))
def did_set_embedding(is_finished):
  if is_finished == None:
    raise PreventUpdate

  if is_finished:
    layers_count = len(Context.embedding.hidden_states)

    options = []

    for i in range(layers_count):
      options.append({'label': 'Layer {}'.format(i), 'value': i})

    return options, SimilarityMethodKey.dash_options()

  return [], []


@app.callback(Output(ElementId.method_confirm_button.value, 'disabled'),
              Input(ElementId.method_layer_dropdown.value, 'value'),
              Input(ElementId.method_dropdown.value, 'value'))
def update_visualize_button(layer, method):
  return layer is None or method is None


@app.callback(Output(ElementId.graph.value, 'figure'),
              Output(ElementId.method_confirm_button.value, 'n_clicks'),
              Input(ElementId.method_confirm_button.value, 'n_clicks'),
              State(ElementId.method_layer_dropdown.value, 'value'),
              State(ElementId.method_dropdown.value, 'value'))
def visualize(clicks, layer, method):
  if clicks != 1 or layer is None or method is None or Context.embedding is None:
    raise PreventUpdate

  embedding = Context.embedding

  figure = go.Figure()

  figure.update_layout(template='ggplot2',
                       xaxis=dict(showgrid=False),
                       yaxis=dict(showgrid=False))

  if method == SimilarityMethodKey.heatmap_plot.value:
    similarity = next(SimilarityMethod.cosine_sim_sentence(embedding, layer))

    figure.add_trace(Plotter.heatmap(similarity, Context.indexed_tokens, Context.indexed_tokens))

    return figure, 0

  if method == SimilarityMethodKey.pca_scatter_plot.value:
    pca = next(SimilarityMethod.pca(embedding, layer))
    normalized = preprocessing.normalize(pca[:, :2])

    figure.add_trace(Plotter.scatter(np.array(normalized), Context.indexed_tokens))

    figure.update_layout(
      xaxis=dict(range=[-2, 2]),
      yaxis=dict(range=[-2, 2])
    )

    return figure, 0

  if method == SimilarityMethodKey.pca_scatter_3d_plot.value:
    pca = next(SimilarityMethod.pca(embedding, layer))
    normalized = preprocessing.normalize(pca[:, :3])

    figure.add_trace(Plotter.scatter_3d(np.array(normalized), Context.indexed_tokens))

    figure.update_layout(
      scene=dict(xaxis=dict(range=[-2, 2]),
                 yaxis=dict(range=[-2, 2]),
                 zaxis=dict(range=[-2, 2]))
    )

    return figure, 0

  raise PreventUpdate


@app.callback(Output(ElementId.method_finish_button.value, 'n_clicks'),
              Input(ElementId.method_finish_button.value, 'n_clicks'))
def finish_flow(clicks):
  if clicks == 1:
    Context.transformer = None
    Context.embedding = None
    Context.indexed_tokens = None

  return 0
