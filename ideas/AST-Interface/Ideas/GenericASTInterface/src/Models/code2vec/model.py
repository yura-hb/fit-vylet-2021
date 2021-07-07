
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

#
# Greetings and thanks for source code
# https://github.com/bentrevett/code2vec/blob/master/models.py
#

class PathEmbedding(pl.LightningModule):

  def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, droupout_prob):
    super().__init__()

    self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)
    self.path_embedding = nn.Embedding(paths_dim, embedding_dim)

    self.weights = nn.Parameter(torch.randn(1, embedding_dim, 3 * embedding_dim))
    self.attention = nn.Parameter(torch.randn(1, embedding_dim, 1))

    self.output = nn.Linear(embedding_dim, output_dim)
    self.dropout = nn.Dropout(droupout_prob)

  def forward(self, start, path, end):
    # Input: start, path, end -> [batch_size, max_length]
    # Output: [batch_size, output_dim]

    # [batch_size, embedding_dim, embedding_dim * 3]
    weights = self.weights.repeat(start[0].shape, 1, 1)

    #
    # Generate embedding vectors for the input
    #

    # embedded -> [batch_size, max_length, embedding_dim]
    embedded_start, embedded_path, embedded_end = self.node_embedding(start), self.node_embedding(path), self.node_embedding(end)

    #
    # Combine embeddings and dropout dimensions from embedding
    #

    # dropped -> [batch_size, max_len, embedding_dim * 3]
    dropped = self.dropout(torch.cat((embedded_start, embedded_path, embedded_end)), dim = 2)
    # dropped -> [batch_size, embedding_dim * 3, max_len]
    dropped = dropped.permute(0, 2, 1)

    #
    # Activation function
    #

    # x -> [batch_size, max_len, embedding_dim]
    x = torch.tanh(torch.bmm(weights, dropped))
    # x -> [batch_size, embedding_dim, max_len]
    x = x.permute(0, 2, 1)

    #
    # Calculate attention vector
    #

    # attention -> [batch_size, embedding_dim, 1]
    attention = self.attention.repeat(start.shape[0], 1, 1)

    #
    # Apply attention vector
    #

    # selected -> [batch_size, max_len]
    selected = torch.bmm(x, attention).squeeze(2)
    # selected -> [batch_size, max_len]
    selected = F.softmax(selected, dim = 1)
    # selected -> [batch_size, max_len, 1]
    selected = selected.unsqueeze(2)

    #
    # Convert to result
    #

    # output -> [batch_size, max_len, embedding_dim]
    output = x.permute(0, 2, 1)
    # output -> [batch_size, embedding_dim]
    output = torch.bmm(x, selected).squeeze(2)
    # output -> [batch_size, output_dim]
    output = self.output(output)

    return output

  def training_step(self):
    # TODO: - Implement
    pass
