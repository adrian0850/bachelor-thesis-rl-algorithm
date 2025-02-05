import itertools
import random
import time

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback, CallbackList
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3 import PPO,DQN

import math
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
import numpy as np


import design_space_funcs as dsf
import FEM as fem
import RL_Env as rl
from feature_extractor import CustomCNN
import constants as const


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0
        self.constraint_list = []

    def _on_step(self) -> bool:
        # Increment current episode length
        self.current_length += 1
        # Add the reward of the current step to the current episode rewards
        self.current_rewards += self.locals['rewards']

        # Check if any episode is done
        if np.any(self.locals['dones']):
            # Append the current episode rewards and length to the lists
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            env = self.training_env
            constraints = env.get_attr("constraint")
            self.constraint_list.append(constraints)
            mean_constraint = np.mean(self.constraint_list[-100:])
            self.logger.record('custom/mean_constraint_value', mean_constraint)
            # Log the mean reward and episode length
            mean_reward = np.mean(self.episode_rewards[-100:])  # Mean reward over the last 100 episodes
            mean_length = np.mean(self.episode_lengths[-100:])  # Mean length over the last 100 episodes
            mean_smartness = mean_reward / mean_length
            self.logger.record('custom/mean_smartness', mean_smartness)

            # Reset current rewards and length
            self.current_rewards = 0
            self.current_length = 0

        return True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def fem_analysis_func(strat, plot):
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY20")]

    # bounded_input = input("Enter bounded coordinates as (row,col) separated by spaces: ")
    # bounded = [tuple(map(int, coord.split(','))) for coord in bounded_input.split()]

    # loaded_input = input("Enter loaded coordinates as (row,col,val) separated by spaces: ")
    # loaded = [tuple(map(int, coord.split(','))) for coord in loaded_input.split()]

    grid = dsf.create_grid(height, width, bounded, loaded)
    print("Grid created:")
    print(grid)

    a,b,c,d = dsf.extract_fem_data(grid)
    print("nodes", a)
    print("element", b)
    # print("bounded:", c)
    # print("loaded:", d)
    fem.plot_mesh(a, b)
    print(len(b))

    # Call the FEM function
    init_vm_stresses, init_avg_strain = fem.FEM(a, b, c, d, plot_flag=plot,
                                                grid=grid, device=const.DEVICE)

    grid =dsf.convert_grid_with_von_mises(grid, init_vm_stresses)

    print(np.round(grid, 3))
    print("np.size(grid)", np.size(grid))
    bad = False
    good = False

    #bad = True
    good = True
    # Modify the grid

    if strat == "bad":
        dsf.remove_material(grid, -2, -1)
        reward, vm_stresses = rl.get_reward(grid, init_avg_strain, plot=plot)
        grid = dsf.convert_grid_with_von_mises(grid, vm_stresses)
        print(np.round(grid[0], 3))
        print("Reward,\t", reward)
        dsf.remove_material(grid, -2, -2)
        reward, vm_stresses = rl.get_reward(grid, init_avg_strain, plot=plot)
        grid = dsf.convert_grid_with_von_mises(grid, vm_stresses)
        print(np.round(grid[0], 3))
        print("Reward,\t", reward)
        dsf.remove_material(grid, -2, -3)
        reward, vm_stresses = rl.get_reward(grid, init_avg_strain, plot=plot)
        grid = dsf.convert_grid_with_von_mises(grid, vm_stresses)
        print(np.round(grid[0], 3))
        print("Reward,\t", reward)
    elif strat == "good":
        dsf.remove_material(grid, 0, -1)
        reward, vm_stresses = rl.get_reward(grid, init_avg_strain, plot=plot)
        grid = dsf.convert_grid_with_von_mises(grid, vm_stresses)
        print(np.round(grid[0], 3))
        print("Reward,\t", reward)
        
        dsf.remove_material(grid, 0, -2)
        reward, vm_stresses = rl.get_reward(grid, init_avg_strain, plot=plot)
        grid = dsf.convert_grid_with_von_mises(grid, vm_stresses)
        print(np.round(grid[0], 3))
        print("Reward,\t", reward)
        
        dsf.remove_material(grid, 1, -1)
        reward, vm_stresses = rl.get_reward(grid, init_avg_strain, plot=plot)
        grid = dsf.convert_grid_with_von_mises(grid, vm_stresses)
        print(np.round(grid[0], 3))
        print("Reward,\t", reward)
    else:
        grid[-0.5*height][-0.5*width] = 0

def env_compatability(plot = False):
    """
    Prompts the user to input grid dimensions, initializes a TopOptEnv environment,
    and checks the environment for compatibility with RL algorithms.
    The function performs the following steps:
    1. Prompts the user to enter the grid width and height.
    2. Initializes the environment with predefined boundary and load conditions.
    3. Checks the environment using the `check_env` function from the RL library.
    Parameters:
    None
    Returns:
    None
    """

    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -0, "LY20")]

    env = rl.TopOptEnv(height, width, bounded, loaded)
    check_env(env, warn=True)


def learn(num_envs, num_timesteps=5e5, model_name="unnamed_model"):
    """
    Runs a reinforcement learning test for topology optimization using PPO (Proximal Policy Optimization).
    This function performs the following steps:
    1. Prompts the user to input the grid width and height.
    2. Defines boundary and load conditions for the environment.
    3. Creates a vectorized environment using SubprocVecEnv.
    4. Sets up a TensorBoard logger for monitoring training progress.
    5. Defines policy keyword arguments for the PPO model.
    6. Creates and trains the PPO model using the specified environment and policy.
    7. Saves the trained model.
    8. Loads the trained model and evaluates it in a new environment.
    9. Runs a loop to predict actions and step through the environment, printing the environment state.
    Note:
        - The environment and model are specific to topology optimization tasks.
        - The model is trained on a CUDA-enabled GPU if available, otherwise on the CPU.
        - The training process logs progress to TensorBoard.
    Inputs:
        - Grid width and height are provided by the user via input prompts.
    Outputs:
        - The trained PPO model is saved to a file named "ppo_topopt".
        - The environment state is printed during evaluation.
    Raises:
        - Any exceptions raised by the underlying libraries (e.g., gym, stable_baselines3, torch) during environment creation, model training, or evaluation.
    """
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY50")]

    # Create the vectorized environment
    env = SubprocVecEnv([rl.make_env(height, width, bounded, loaded) for _ in range(num_envs)])
    env = VecMonitor(env)
    #env = rl.TopOptEnv(height, width, bounded, loaded)
    # Set up TensorBoard logger
    log_dir = "./tensorboard_logs/total strain/PPO_custom_cnngamma0.9_5x5 general thresh 0.67/"

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256)
    )
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
    logging_callback = CustomLoggingCallback()

    callback_list = CallbackList([eval_callback, logging_callback])
    logger = configure(log_dir, ["tensorboard"])
    # # Create the PPO model with the logger

    ########DQN WITH CUSTOM CNN
    # model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, 
    #             learning_rate= 2.5e-3, batch_size=64,
    #             policy_kwargs=policy_kwargs,device=const.DEVICE)


    # ########DQN WITH standard POLICY
    # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
    #             learning_rate= 2.5e-3, batch_size=64, exploration_initial_eps= 1,
    #             exploration_fraction=0.4, device=const.DEVICE)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, 
                learning_rate= 2.5e-4, n_steps=1000, batch_size=500, n_epochs=10, 
                policy_kwargs=policy_kwargs, gae_lambda=0.95, gamma=0.9,
                device=const.DEVICE)
    
    
    #model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir,
                # learning_rate= 2.5e-3, n_steps=100, batch_size=64, n_epochs=10,
                # gae_lambda=0.95, device=const.DEVICE)

    # Train the model
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    #     with record_function("model_training"):
    #         # Train the model
    #         model.learn(total_timesteps=num_timesteps, progress_bar=True)

    model.set_logger(logger)
    model.learn(total_timesteps=num_timesteps, progress_bar=True, callback=callback_list, reset_num_timesteps=False)
    model.save(model_name)
    
 




    env.close()
    env = rl.TopOptEnv(height, width, bounded, loaded, mode="eval")
    model = PPO.load(model_name, env = env)
    obs, _ = env.reset()
    for i in range(10):
        action, _states = model.predict(obs)
        action = action.item()
        print("Action: ", env.action_to_coordinates(action))
        obs, rewards, dones, _, info = env.step(action)
        env.print()

def test_all_grids():
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY10")]

    env = rl.TopOptEnv(height, width, bounded, loaded, mode="eval")

    # Generate all possible action sequences
    num_actions = width * height
    actions = [i for i in range(num_actions) if i not in [0, 12, 15]]
    action_sequences = itertools.permutations(actions, (len(actions) // 2))
    obs_list = []
    index = 0
    for action_sequence in action_sequences:
    
        if 0  not in action_sequence or 12 not in action_sequence or 15 not in action_sequence:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            for action in action_sequence:
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                if trunc is True:
                    obs_tuple = tuple(obs.flatten())
                    if obs_tuple not in obs_list:
                        #print (obs[0,:,:])
                        print(f"{index} Action sequence: {action_sequence}, Total reward: {total_reward}")
                        env.print()
                        obs_list.append(obs_tuple)
                        index += 1
                if done:
                    break


        # Check if the volume fraction is reached
 



def load(model_name = "unnamed_model"):
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))
    start_time_1 = time.time()
    bounded = [(0, 0), (-1,0 )]
    loaded = [(-1, -1, "LY10")]

    env = rl.TopOptEnv(height, width, bounded, loaded, mode="eval")
    model = PPO.load(model_name, env = env)

    dones = False
    obs, _ = env.reset()
    end_time_1 = time.time()
    print(obs)
    start_time_2 = time.time()
    while dones != True:

        action, _states = model.predict(obs)
        action = action.item()
        print("Action: ", env.action_to_coordinates(action))
        print(divmod(action, width))
        obs, rewards, dones, _, info = env.step(action)
        print("Reward = ", rewards)
        print(obs)

        env.print()
    end_time_2 = time.time()
    print("Time taken to load model: ", end_time_1 - start_time_1)
    print("Time taken to run model: ", end_time_2 - start_time_2)

def loading_learn(num_envs, num_timesteps, model_name = "unnamed_model"):
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY50")]

    env = SubprocVecEnv([rl.make_env(height, width, bounded, loaded) for _ in range(num_envs)])

    env = VecMonitor(env)

    
    #env = rl.TopOptEnv(height, width, bounded, loaded)
    # Set up TensorBoard logger
    log_dir = "./tensorboard_logs/total strain/PPO_custom_cnngamma0.9_5x5 general thresh 0.67/"



    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
    logging_callback = CustomLoggingCallback()

    callback_list = CallbackList([eval_callback, logging_callback])
    logger = configure(log_dir, ["tensorboard"])
    # Create the PPO model with the logger
    model = PPO.load(model_name, env = env)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
    # Train the model

    model.set_logger(logger)
    model.learn(total_timesteps=num_timesteps, progress_bar=True, callback=callback_list, reset_num_timesteps=False)
    model.save(model_name)




    obs, _ = env.reset()
    for i in range(20):
        action, _states = model.predict(obs)
        #action = action.item()
        print(divmod(action, width))
        print("Action: ", env.action_to_coordinates(action))
        obs, rewards, dones, _, info = env.step(action)
        env.print()

def cnn():
    set_seed(42)
    # Get grid dimensions from user input
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    # Define bounded and loaded points
    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY50")]

    # Instantiate your actual topology optimization environment
    env = rl.make_env(height, width, bounded, loaded, mode="eval")()

    # Instantiate the feature extractor
    feature_extractor = fe.CustomCombinedExtractor(env.observation_space)

    device = const.DEVICE


    print("CNN Structure:")
    for key, extractor in feature_extractor.extractors.items():
        print(f"Extractor for key: {key}")
        extractor.to(device)
        summary(extractor, input_size=env.observation_space[key].shape, device=device)

    # Get a sample observation
    sample_observation, _ = env.reset()
    print(env.grid)
    #Convert the sample observation to a PyTorch tensor
    sample_observation_tensor = {
        key: torch.tensor(value).unsqueeze(0) for key, value in sample_observation.items()
    }
    # print(sample_observation_tensor)

    # Pass the sample observation through the feature extractor
    features = feature_extractor(sample_observation_tensor)
    feature_extractor.to(device)

    # Print the extracted features
    print("Extracted Features Shape:", features.shape)
    print("Extracted Features:", features)

    feature_extractor.visualize_feature_maps()

    # Check value ranges
    min_val = features.min().item()
    max_val = features.max().item()
    print("Min value in features:", min_val)
    print("Max value in features:", max_val)

    # Plot histogram of feature values
    features_np = features.detach().cpu().numpy().flatten()
    plt.hist(features_np, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Extracted Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.show()

    # Visualize features as an image
    # Reshape the features to a 2D array for visualization
    features_np = features.detach().cpu().numpy().reshape(1, -1)
    plt.imshow(features_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Extracted Features Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Batch Index")
    plt.show()
