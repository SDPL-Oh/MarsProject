import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_agent(env, episodes=500, gamma=0.99, lr=1e-3,
                batch_size=64, buffer_size=10000, epsilon_start=1.0,
                epsilon_end=0.05, epsilon_decay=0.995):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQNAgent(state_dim, action_dim).to(device)
    target_net = DQNAgent(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = deque(maxlen=buffer_size)

    epsilon = epsilon_start

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(200):  # max steps per episode
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # epsilon-greedy 정책
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            # 환경에 적용
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # 버퍼 저장
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            # 학습
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Q(s,a)
                q_values = policy_net(states).gather(1, actions)

                # Q_target = r + gamma * max_a' Q_target(s',a')
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    q_targets = rewards + gamma * max_next_q * (1 - dones)

                loss = nn.MSELoss()(q_values.squeeze(), q_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if ep % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {ep+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return policy_net
