import torch


class ContrastiveEncoder(torch.nn.Module):
    def __init__(self, state_dimension, hidden_size=128, feature_size=32):
        super().__init__()
        self.linear1 = torch.nn.Linear(state_dimension, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        y = self.linear1(x).clamp(min=0)
        y = self.linear2(y).clamp(min=0)
        y = self.linear3(y)
        return y


class ContrastiveFutureState(torch.nn.Module):
    def __init__(self, state_dimension, hidden_size=128, feature_size=32):
        super().__init__()
        self.encoder = ContrastiveEncoder(state_dimension, hidden_size=hidden_size, feature_size=feature_size)
        self.W = torch.nn.Parameter(torch.rand(feature_size, feature_size))

    def encode(self, x):
        return self.encoder(x)

    def forward(self, s_0, s_h):
        z_a = self.encode(s_0)
        z_pos = self.encode(s_h)
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
