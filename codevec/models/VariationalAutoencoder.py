
from pytorch_lightning import LightningModule
from dataclasses import dataclass

import torch
from torch import nn

from typing import Tuple, Any

class VariationalAutoEncoder(LightningModule):

  @dataclass
  class Config:
    encoder: nn.Sequential
    decoder: nn.Sequential
    encoder_output_dim: int

    latent_dim: int = 12
    learning_rate: float = 0.02

  def __init__(self, config: Config):
    super().__init__()

    self.save_hyperparameters()

    self.config = config

    self.encoder = self.config.encoder
    self.decoder = self.config.decoder

    self.mu = nn.Linear(self.config.encoder_output_dim, self.config.latent_dim)
    self.var = nn.Linear(self.config.encoder_output_dim, self.config.latent_dim)

    self.log_scale = nn.Parameter(torch.Tensor([0.0]))

  def training_step(self, batch, batch_idx):
    x, _ = batch
    z, mu, _, std = self.get_latent_encoding(x)

    decoded = self.decoder(z)

    reconstruction_loss = self.__gaussian_likehood(decoded, self.log_scale, x)
    kl_divergence = self.__kl_divergence(z, mu, std)

    elbo = kl_divergence - reconstruction_loss
    elbo = elbo.mean()

    self.log('training_elbo_loss', elbo, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return elbo

  def get_latent_encoding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor]:
    encoded = self.encoder(x)
    mu, var = self.mu(encoded), self.var(encoded)

    std = torch.exp(var / 2)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()

    return z, mu, var, std

  def configure_optimizers(self):
    # (self.lr or self.config.learning_rate) enables automatic lr finding from paper
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html
    return torch.optim.Adam(self.parameters(), lr=(self.lr or self.config.learning_rate))

  def predict_step(self, batch: Any, batch_idx: int) -> Any:
    x, _ = batch

    return self.get_latent_encoding(batch)

  @staticmethod
  def __kl_divergence(z, mu, std):
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)

    return kl

  @staticmethod
  def __gaussian_likelihood(mean, log_scale, sample):
    scale = torch.exp(log_scale)
    distribution = torch.distributions.Normal(mean, scale)
    log_pxz = distribution.log_prob(sample)

    return log_pxz.sum()