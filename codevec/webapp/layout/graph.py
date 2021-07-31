import dash_core_components as dcc
import dash_html_components as html

from utils.constants import ElementId

layout = html.Div([dcc.Graph(id=ElementId.graph.value, responsive=True, className=ElementId.graph.value)],
           className=ElementId.graph_container.value)
