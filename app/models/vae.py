import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, seq_len=20, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 32), nn.ReLU(),
            nn.Linear(32, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, seq_len), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def load_vae_model(path: str) -> VAE:
    model = VAE()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
