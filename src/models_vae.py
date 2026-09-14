"""
Class-specific convolutional VAE, matching the project spec:
  Encoder: 3->32->64->128->256, each 4x4 stride-2 conv + BatchNorm + SiLU
  Latent: dim 128, two linear heads for mu / logvar
  Decoder: linear -> 256x8x8 -> transposed-conv 256->128->64->32->3, sigmoid
  Assumes 128x128 input (8x8 spatial after 4 stride-2 downsamples).
"""
import torch
import torch.nn as nn

from config import VAE_IMAGE_SIZE, VAE_LATENT_DIM

_DOWNSAMPLE_STEPS = 4
_FEATURE_SPATIAL = VAE_IMAGE_SIZE // (2 ** _DOWNSAMPLE_STEPS)  # 128 / 16 = 8


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = VAE_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        def enc_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            enc_block(3, 32),
            enc_block(32, 64),
            enc_block(64, 128),
            enc_block(128, 256),
        )
        flat_dim = 256 * _FEATURE_SPATIAL * _FEATURE_SPATIAL
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flat_dim)

        def dec_block(cin, cout, final=False):
            layers = [nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1)]
            if not final:
                layers += [nn.BatchNorm2d(cout), nn.SiLU(inplace=True)]
            return nn.Sequential(*layers)

        self.decoder = nn.Sequential(
            dec_block(256, 128),
            dec_block(128, 64),
            dec_block(64, 32),
            dec_block(32, 3, final=True),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 256, _FEATURE_SPATIAL, _FEATURE_SPATIAL)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z).clamp(0.0, 1.0)


def vae_loss(x_hat, x, mu, logvar, beta):
    """L = reconstruction L1 + beta * KL(q(z|x) || N(0,I)), matching the spec."""
    recon = torch.nn.functional.l1_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon.detach(), kl.detach()


def beta_schedule(epoch: int, warmup_epochs: int, beta_start: float, beta_end: float):
    if warmup_epochs <= 0:
        return beta_end
    frac = min(1.0, epoch / warmup_epochs)
    return beta_start + frac * (beta_end - beta_start)
