import gymnasium as gym
import torch as th
import matplotlib.pyplot as plt
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from RL_Env import TopOptEnv

k_size = 3

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=k_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=k_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=k_size, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Adaptive pooling to ensure consistent output size
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    def visualize_feature_maps(self, observations: th.Tensor):
        x = observations
        for layer in self.cnn:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.plot_feature_maps(x)
        
        # Visualize the final feature vector
        final_features = self.linear(x)
        self.plot_final_features(final_features)

    def plot_feature_maps(self, feature_maps):
        num_feature_maps = feature_maps.shape[1]
        size = feature_maps.shape[2]
        fig, axes = plt.subplots(num_feature_maps, 1, figsize=(size, num_feature_maps * 2))
        for i in range(num_feature_maps):
            axes[i].imshow(feature_maps[0, i].detach().cpu().numpy(), cmap='viridis')
            axes[i].axis('off')
        plt.show()

    def plot_final_features(self, final_features):
        final_features = final_features.detach().cpu().numpy()
        plt.figure(figsize=(10, 2))
        plt.imshow(final_features, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Final Feature Vector")
        plt.show()

def main():
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY20")]

    # Create the environment
    env = TopOptEnv(height, width, bounded, loaded, mode="eval")
    obs, _ = env.reset()
    print(obs)
    env.print()
    obs = th.as_tensor(obs[None]).float()

    feature_extractor = CustomCNN(env.observation_space)
    feature_extractor.visualize_feature_maps(obs)

if __name__ == "__main__":
    main()