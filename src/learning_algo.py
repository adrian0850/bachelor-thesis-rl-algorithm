import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np
from RL_Env import TopOptEnv

# Constants
NUM_ACTIONS = 6 * 12  # Example value, replace with the actual number of actions
STATE_SHAPE = (4, 12, 6)  # Example state shape

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        
        # Calculate the output size of the convolutional layers
        conv_output_size = 4 * 12 * 6  # 4 channels, 12 height, 6 width
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DoubleQTrainer:
    def __init__(self, env, num_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, batch_size=32, memory_size=2000):
        self.env = env
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.main_model = QNetwork()
        self.auxiliary_model = QNetwork()
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self.main_model(torch.FloatTensor(state).unsqueeze(0))
        return q_values.argmax().item()

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_action = self.main_model(torch.FloatTensor(next_state).unsqueeze(0)).argmax().item()
                target += self.gamma * self.auxiliary_model(torch.FloatTensor(next_state).unsqueeze(0))[0][next_action].item()
            target_f = self.main_model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.main_model(torch.FloatTensor(state).unsqueeze(0)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_auxiliary_model(self):
        self.auxiliary_model.load_state_dict(self.main_model.state_dict())

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.train_model()
            self.update_auxiliary_model()

# Example usage
width = int(input("Enter grid width: "))
height = int(input("Enter grid height: "))

bounded = [(0, 0), (-1, 0)]
loaded = [(-1, -0, "LY20")]

env = TopOptEnv(width, height, bounded, loaded)
trainer = DoubleQTrainer(env, NUM_ACTIONS)
trainer.train(1000)