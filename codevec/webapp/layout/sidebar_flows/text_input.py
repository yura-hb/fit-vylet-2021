import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from utils.constants import ElementId

from app import app, Context

layout = html.Div([
  html.H2("2. Input text",
          id=ElementId.input_label.value,
          className=ElementId.input_label.value),

  html.Div([], className=ElementId.input_separator.value),

  dcc.Textarea(id=ElementId.input_textarea.value,
               className=ElementId.input_textarea.value,
               placeholder="Enter your text "),

  html.H6("Tokens: None", id=ElementId.input_token_info.value, className=ElementId.input_token_info.value),
  html.H6("Batches: None", id=ElementId.input_batch_info.value, className=ElementId.input_batch_info.value),

  html.Div([
    dbc.Button("Visualize",
               id=ElementId.input_confirm_button.value,
               color="primary",
               className="mr-1",
               block=True,
               href=ElementId.analysis_path.value)
  ], className=ElementId.input_confirm_button.value)
])


@app.callback(Output(ElementId.input_token_info.value, 'children'),
              Output(ElementId.input_batch_info.value, 'children'),
              Input(ElementId.input_textarea.value, 'value'))
def update_tokens(text):
  if Context.transformer == None or text is None:
    raise PreventUpdate

  if text == '':
    return 'Tokens: {}'.format(0), 'Batches: {}'.format(0)

  tokenized = Context.transformer.tokenize(text)

  return 'Tokens: {}'.format(tokenized.input_ids.shape[1]), 'Batches: {}'.format(tokenized.input_ids.shape[0])


@app.callback(Output(ElementId.input_confirm_button.value, 'n_clicks'),
              Output(ElementId.is_embedding_ready.value, 'data'),
              Input(ElementId.input_confirm_button.value, 'n_clicks'),
              State(ElementId.input_textarea.value, 'value'))
def generate_embedding(clicks, text):
  if Context.transformer == None:
    raise PreventUpdate

  if clicks:
    tokenized = Context.transformer.tokenize(text)
    Context.embedding = Context.transformer(tokenized)

    tokens = Context.transformer.tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])

    Context.indexed_tokens = ['{}_{}'.format(id, token) for id, token in enumerate(tokens)]

    return 0, True

  raise PreventUpdate
