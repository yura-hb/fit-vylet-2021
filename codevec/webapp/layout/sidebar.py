import dash_html_components as html

from utils.constants import ElementId
from dash.dependencies import Input, Output, State
from app import app

import dash

from .sidebar_flows.model import layout as model_layout
from .sidebar_flows.method import layout as method_layout
from .sidebar_flows.text_input import layout as text_input_layout

layout = html.Div([
  html.Button(
    html.I(className="fa fa-bars"),
    id=ElementId.sidebar_toggle.value,
    className=ElementId.sidebar_toggle.value
  ),
  html.Div([],
           id=ElementId.sidebar_content_container.value,
           className=ElementId.sidebar_content_container.value)
], id=ElementId.sidebar.value, className=ElementId.sidebar.value)

@app.callback(Output(ElementId.sidebar_toggle.value, "n_clicks"),
              Output(ElementId.sidebar.value, "className"),
              Output(ElementId.sidebar_content_container.value, "className"),
              [Input(ElementId.sidebar_toggle.value, "n_clicks")],
              State(ElementId.sidebar.value, 'className'))
def toggle_sidebar(clicks, className):
  if clicks == None:
    return 0, ElementId.sidebar.value, ElementId.sidebar_content_container.value

  if className == ElementId.sidebar.value:
    return 0, ElementId.sidebar_collapsed.value, ElementId.sidebar_content_container_hidden.value

  return 0, ElementId.sidebar.value, ElementId.sidebar_content_container.value

@app.callback(Output(ElementId.sidebar_content_container.value, 'children'),
              Input(ElementId.is_active_flow.value, 'data'),
              State(ElementId.url.value, 'pathname'))
def update_sidebar_content(is_active, path):
  if not is_active:
    return dash.no_update

  if is_active:
    if path == ElementId.model_input_path.value:
      return [model_layout]

    if path == ElementId.text_input_path.value:
      return [text_input_layout]

    if path == ElementId.analysis_path.value:
      return [method_layout]

  return dash.no_update
