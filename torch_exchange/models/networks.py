import torch
import torch.nn as nn
from torch.distributions import Normal

HIDDEN_DIM = 256

class CNNEncoder(nn.Module):
    def __init__(self, n_stack, input_h, input_w):
        super().__init__()
        
        # Input: (N, C, H, W) where C=n_stack, H=input_h (depth), W=input_w (4)
        # We want to convolve over Price Levels (H) and Time (C)
        # But usually FrameStack is channel dim.
        
        self.net = nn.Sequential(
            nn.Conv2d(n_stack, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 4), padding=(1, 0)), # Convolve over full width (4 features)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_h * 1, HIDDEN_DIM),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (Batch, Stack*Depth*4) -> Reshape to (Batch, Stack, Depth, 4)
        # Note: We need to know Stack and Depth.
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, depth, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(4, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, depth, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, HIDDEN_DIM)
        
    def forward(self, x):
        # x shape: (Batch, Depth, 4)
        B, D, _ = x.shape
        x = self.embedding(x) + self.pos_encoder[:, :D, :]
        x = self.transformer(x)
        # Mean pooling
        x = x.mean(dim=1)
        x = torch.relu(self.fc(x))
        return x

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, model_type='mlp', book_depth=10, n_stack=1):
        super(ActorCritic, self).__init__()
        self.model_type = model_type
        self.book_depth = book_depth
        self.n_stack = n_stack
        
        if model_type == 'cnn':
             # Assumes obs is flattened (Stack * Depth * 4)
             # We need to reshape inside forward
             self.encoder = CNNEncoder(n_stack, book_depth, 4)
             
        elif model_type == 'transformer':
             # Assumes obs is flattened (Depth * 4) -> (Batch, Depth, 4)
             self.encoder = TransformerEncoder(book_depth)
             
        else: # mlp
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, HIDDEN_DIM),
                nn.Tanh(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.Tanh()
            )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Tanh(),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1)
        )
        
    def forward(self, x):
        if self.model_type == 'cnn':
            # Reshape x: (B, Stack*Depth*4) -> (B, Stack, Depth, 4)
             B = x.shape[0]
             x = x.view(B, self.n_stack, self.book_depth, 4)
             
        elif self.model_type == 'transformer':
            # Reshape x: (B, Depth*4) -> (B, Depth, 4)
            B = x.shape[0]
            x = x.view(B, self.book_depth, 4)
            
        features = self.encoder(x)
        return features
        
    def get_action_and_value(self, x, action=None):
        features = self.forward(x)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        action_log_probs = probs.log_prob(action).sum(axis=-1)
        dist_entropy = probs.entropy().sum(axis=-1)
        value = self.critic(features)
        
        return action, action_log_probs, dist_entropy, value
