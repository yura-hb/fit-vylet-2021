
from enum import Enum


class Model(Enum):
  bert_base_cased = 'bert-base-cased'
  gpt2 = 'gpt2'

  @staticmethod
  def dash_options():
    return [
      {'label': 'Bert base cased', 'value': Model.bert_base_cased.value},
      {'label': 'GPT-2', 'value': Model.gpt2.value},
    ]


class SimilarityMethodKey(Enum):
  heatmap_plot = 'heatmap'
  pca_scatter_plot = 'pca'
  pca_scatter_3d_plot = 'pca_3d'

  @staticmethod
  def dash_options():
    return [
      {'label': 'Cosine similarity', 'value': SimilarityMethodKey.heatmap_plot.value},
      {'label': 'PCA 2d', 'value': SimilarityMethodKey.pca_scatter_plot.value},
      {'label': 'PCA 3d', 'value': SimilarityMethodKey.pca_scatter_3d_plot.value}
    ]


class ElementId(Enum):
  url = 'url'
  app_container = 'app-container'
  graph_container = 'graph-container'

  model_store = 'model-store'
  is_embedding_ready = 'is-embedding-ready'

  #
  # Method selector view
  #
  method_label = 'method-label'
  method_separator = 'method-separator'
  method_layer_dropdown = 'method-layer-dropdown'
  method_dropdown = 'method-dropdown'
  method_confirm_button = 'method-confirm-button'
  method_finish_button = 'method-finish-button'

  graph = 'graph'

  #
  # Input view
  #
  input_label = 'input-label'
  input_separator = 'input-separator'
  input_textarea = 'input-textarea'
  input_token_info = 'input-token-info'
  input_batch_info = 'input-batch-info'
  input_confirm_button = 'input-confirm-button'

  #
  # Sidebar
  #
  sidebar = 'sidebar'
  sidebar_collapsed = 'sidebar-collapsed'
  sidebar_content_container = 'sidebar-content-container'
  sidebar_content_container_hidden = 'sidebar-content-container-hidden'
  sidebar_toggle = 'sidebar-toggle'

  #
  # Model selector view
  #
  model_label = 'model-label'
  model_separator = 'model-separator'
  model_dropdown = 'model-dropdown'
  model_confirm_button = 'model-confirm-button'
  model_finish_loading_modal_id = 'model-finish-loading-modal'
  model_finish_loading_modal_text = 'model-finish-loading-modal-text'
  model_finish_loading_modal_confirm_button = 'model-finish-loading-modal-confirm-button'
  model_finish_loading_modal_cancel_button = 'model-finish-loading-modal-cancel-button'

  #
  # Helper flags
  #
  is_active_flow = 'is-active-screen'

  #
  # Paths
  #
  model_input_path = '/app/model'
  text_input_path = '/app/input'
  analysis_path = '/app/analysis'
