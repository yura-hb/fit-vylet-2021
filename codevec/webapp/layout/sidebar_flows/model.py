import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from utils.constants import ElementId, Model
from app import app, Context

from dash.exceptions import PreventUpdate
import dash

from codevec.models import Transformer

layout = html.Div([
  html.H2("1. Select Model",
          id=ElementId.model_label.value,
          className=ElementId.model_label.value),
  html.Div([], className=ElementId.model_separator.value),
  html.Div([dcc.Dropdown(id=ElementId.model_dropdown.value,
                         options=Model.dash_options(),
                         value=None)], className=ElementId.model_dropdown.value),
  html.Div([
    dbc.Button("Load model", id=ElementId.model_confirm_button.value,
               color="primary", className="mr-1", block=True)
  ], className=ElementId.model_confirm_button.value),

  dbc.Modal([
    dbc.ModalHeader('Completed'),
    dbc.ModalBody('', id=ElementId.model_finish_loading_modal_text.value),
    dbc.ModalFooter([dbc.Button("Add input", id=ElementId.model_finish_loading_modal_confirm_button.value,
                                color="primary", className="mr-1", block=True, href=ElementId.text_input_path.value),
                     dbc.Button("Cancel", id=ElementId.model_finish_loading_modal_cancel_button.value,
                                color="danger", className="mr-1", block=True)])
  ], id=ElementId.model_finish_loading_modal_id.value,
     is_open=False)
])


@app.callback(Output(ElementId.model_confirm_button.value, 'disabled'),
              Input(ElementId.model_dropdown.value, 'value'))
def did_select_model(model):
  return model == None


@app.callback(Output(ElementId.model_finish_loading_modal_id.value, 'is_open'),
              Output(ElementId.model_finish_loading_modal_text.value, 'children'),
              Output(ElementId.model_confirm_button.value, 'n_clicks'),
              Output(ElementId.model_finish_loading_modal_cancel_button.value, 'n_clicks'),

              [Input(ElementId.model_confirm_button.value, 'n_clicks'),
               Input(ElementId.model_finish_loading_modal_cancel_button.value, 'n_clicks')],

              State(ElementId.model_dropdown.value, 'value'))
def load_model(confirm_clicks, cancel_clicks, model_path):
  user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

  if user_click == ElementId.model_confirm_button.value and confirm_clicks == 1:
    print('Model did changed')

    config = Transformer.Config(model_path,
                                model_path,
                                True,
                                model_args={'output_hidden_states': True},
                                autograd=False)

    Context.transformer = Transformer(config)

    return True, "{}".format(Context.transformer.model.config), 0, 0

  if user_click == ElementId.model_finish_loading_modal_cancel_button.value and cancel_clicks == 1:
    Context.transformer = None

    return False, "", 0, 0

  raise PreventUpdate
