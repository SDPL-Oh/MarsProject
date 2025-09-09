import torch
import torch.nn as nn

class SimpleRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_agent(env, agent, episodes=100):
    # 강화학습 학습 루프 (PPO, DQN 등으로 확장 가능)
    pass
