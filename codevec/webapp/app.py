import dash

from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc

inline = False

app = None

class Context:
  transformer = None
  embedding = None
  indexed_tokens = None

if inline:
  app = JupyterDash(__name__,
                    external_stylesheets=[dbc.themes.BOOTSTRAP,
                                          "https://use.fontawesome.com/releases/v5.8.1/css/all.css"],
                    suppress_callback_exceptions=True,
                    meta_tags=[
                        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                    ])
else:
  app = dash.Dash(__name__,
                  external_stylesheets=[dbc.themes.BOOTSTRAP,
                                        "https://use.fontawesome.com/releases/v5.8.1/css/all.css"],
                  suppress_callback_exceptions=True,
                  meta_tags=[
                      {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                  ])
