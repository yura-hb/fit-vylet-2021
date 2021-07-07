
import torch
import torch.nn as nn

class BertEncoder:

  _max_model_input = 512

  def __init__(self, tokenizer, model):
    super().__init__()

    self.tokenizer = tokenizer
    self.model = model

  def encode(self, text):
    """
    Encodes text to format, which can be consumed by the BERT model
    """
    tokens = self.tokenizer.encode_plus(
      text,
      add_special_tokens = False,
      return_tensors = 'pt'
    )

    input_ids_chunks = list(tokens['input_ids'][0].split(self._max_model_input - 2))
    mask_chunks = list(tokens['attention_mask'][0].split(self._max_model_input - 2))

    for i in range(len(input_ids_chunks)):
      # Add CLS and SEP token for BERT
      input_ids_chunks[i] = torch.cat([
         torch.tensor([101]), input_ids_chunks[i], torch.tensor([102])
        ])
      mask_chunks[i] = torch.cat([
        torch.tensor([1]), mask_chunks[i], torch.tensor([1])
      ])

      # Pad to length of 512
      padding_length = self._max_model_input - input_ids_chunks[i].shape[0]

      if padding_length > 0:
        input_ids_chunks[i] = torch.cat([
          input_ids_chunks[i], torch.zeros(padding_length)
        ])

        mask_chunks[i] = torch.cat([
          mask_chunks[i], torch.zeros(padding_length)
        ])

    # Reshape for BERT
    input_ids = torch.stack(input_ids_chunks)
    mask = torch.stack(mask_chunks)

    return {
      'input_ids': input_ids.long(),
      'attention_mask': mask.int()
    }

  def calculate_input_vector(self, lhs_embeddings, rhs_embeddings):
    assert lhs_embeddings.shape == rhs_embeddings.shape, \
           "Embeddings should have the same length"
    assert (lhs_embeddings.shape[0] <= self._max_model_input and
            rhs_embeddings.shape[0] <= self._max_model_input), \
            "Embeddings vectors shouldn't exceed max bert input (512)"
    # Interaction:

    # [768, lhs_len] @ [rhs_len, 768] = [768, 768]
    attention = lhs_embeddings.T @ rhs_embeddings

    # [736, 736]
    softmax_lhs = nn.functional.softmax(attention, dim = 1) # Softmax by row
    softmax_rhs = nn.functional.softmax(attention, dim = 0) # Softmax by column

    # [736, 736]
    weighted_lhs = (softmax_lhs * torch.sum(rhs_embeddings, dim = 0))[:lhs_embeddings.shape[0]]
    weighted_rhs = (softmax_rhs * torch.sum(lhs_embeddings, dim = 0))[:rhs_embeddings.shape[0]]

    # Aggregation:

    # Get max by row
    pooled_lhs = nn.functional.max_pool1d(lhs_embeddings,
                                          kernel_size=lhs_embeddings.shape[1])
    pooled_rhs = nn.functional.max_pool1d(rhs_embeddings,
                                          kernel_size=rhs_embeddings.shape[1])
    pooled_weighted_lhs = nn.functional.max_pool1d(weighted_lhs,
                                                   kernel_size=weighted_lhs.shape[1])
    pooled_weighted_rhs = nn.functional.max_pool1d(weighted_rhs,
                                                   kernel_size=weighted_rhs.shape[1])

    concatenated_lhs = torch.cat([pooled_lhs,
                                  pooled_weighted_lhs,
                                  torch.abs(pooled_lhs - pooled_weighted_lhs)])

    concatenated_rhs = torch.cat([pooled_rhs,
                                  pooled_weighted_rhs,
                                  torch.abs(pooled_weighted_rhs)])

    result = torch.cat([concatenated_lhs,
                        concatenated_rhs,
                        torch.abs(concatenated_lhs - concatenated_rhs)])

    return result

  def get_embeddings(self, tokens):
    """
    Extracts embedding from the tokens
    """
    embeddings = []

    with torch.no_grad():
      output = model(**tokens)

      hidden_states = output.hidden_states

      # [layer, batch, token, embedding]
      embeddings = torch.stack(hidden_states, dim = 0)

      # [layer, token, embedding]
      embeddings = embeddings.squeeze(dim = 1)

      # [token, layer, embedding]
      embeddings = embeddings.permute(1, 0, 2)

      # [token, embedding]
      embeddings = embeddings[:, 0, :]

    return embeddings
