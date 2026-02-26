import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim, dropout_p=0.1):
        super().__init__()
        
        # Main path
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, latent_dim)
        )
        
        
        self.skip_path = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        
        return self.main_path(x) + self.skip_path(x)

class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim, dropout_p=0.1):
        super().__init__()
        
        # Main path
        self.main_path = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, output_dim)
        )
        

        self.skip_path = nn.Linear(latent_dim, output_dim)

    def forward(self, z):

        return self.main_path(z) + self.skip_path(z)

class DAE(nn.Module):

    def __init__(self, input_dim, latent_dim, noise_factor=0.1, dropout_p=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor
        
        self.encoder = Encoder(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            dropout_p=dropout_p
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim, 
            output_dim=input_dim, 
            dropout_p=dropout_p
        )

    def forward(self, x):
        if self.training:
            noise = self.noise_factor * torch.randn_like(x)
            x_noisy = x + noise
        else:
            x_noisy = x
            

        z = self.encoder(x_noisy)
        

        x_recon = self.decoder(z)
        
        return x_recon , z
