
from app import app

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from utils.constants import ElementId

from dash.dependencies import Input, Output
from layout import graph, sidebar

app.layout = html.Div([
  html.Div([], id=ElementId.app_container.value, className=ElementId.app_container.value),
  dcc.Location(id=ElementId.url.value, refresh=False, pathname=ElementId.model_input_path.value),

  dcc.Store(id=ElementId.is_active_flow.value, data=False),
  dcc.Store(id=ElementId.is_embedding_ready.value, data=False)
])


@app.callback(Output(ElementId.app_container.value, 'children'),
              Output(ElementId.is_active_flow.value, 'data'),
              Input(ElementId.url.value, 'pathname'))
def route(pathname):
  if pathname == ElementId.model_input_path.value:
    return [sidebar.layout], True

  if pathname == ElementId.text_input_path.value:
    return [sidebar.layout], True

  if pathname == ElementId.analysis_path.value:
    return [sidebar.layout, graph.layout], True

  return dbc.Jumbotron([
    html.H1("404: Not found", className="text-danger"),
    html.Hr(),
    html.P(f"The pathname {pathname} was not recognised...")]), False


if __name__ == '__main__':
  app.run_server(debug=True)
