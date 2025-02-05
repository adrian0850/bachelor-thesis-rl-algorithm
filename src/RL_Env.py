import os
import design_space_funcs as dsf
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import constants as const
import random
import torch

import time

import FEM as fem



def old_reward_function(design, init_stress, curr_stress):
    # Calculate the ratio of initial to current number of elements
    initial_num_elements = np.size(design)
    current_num_elements = np.count_nonzero(design[0, :, :])

    element_ratio = (initial_num_elements / current_num_elements)
    w_max_stress = 4
    w_max_strain = 4
    # Calculate the ratios of initial to current stress and strain values
    stress_ratio = (initial_max_stress / current_max_stress) * w_max_stress + (initial_avg_stress / current_avg_stress)
    strain_ratio = (initial_max_strain / current_max_strain) * w_max_strain + (initial_avg_strain / current_avg_strain)

    # Combine the ratios and square the result
    reward = (stress_ratio + strain_ratio) ** 2

    reward = 100*reward / (10 + reward)
    return reward

def reward_function(grid, init_strain, curr_strain):
    initial_num_elements = np.size(grid[const.DESIGN, :, :])
    current_num_voided = initial_num_elements - np.count_nonzero(grid[const.DESIGN, :, :])
    element_ratio = ((current_num_voided / initial_num_elements)**2) *4
    strain_ratio = (init_strain / curr_strain)**2
    reward = element_ratio + strain_ratio
    # reward = strain_ratio
    # print(curr_strain)
    return reward

def get_reward(grid, init_strain, plot=False):
    a,b,c,d = dsf.extract_fem_data(grid)
    von_mises_stresses, curr_avg_strain = fem.FEM(a, b, c, d, plot_flag=plot,
                                         grid=grid, device=const.DEVICE)
    reward = reward_function(grid, init_strain, curr_avg_strain)
    return reward, von_mises_stresses

def get_needed_fem_values(grid, plot=False):
    a,b,c,d = dsf.extract_fem_data(grid)
    von_mises_stresses, curr_avg_strain = fem.FEM(a, b, c, d, plot_flag=plot,
                                         grid=grid, device=const.DEVICE)
    return von_mises_stresses, curr_avg_strain

def get_observation_space(height, width):
    low = np.zeros((height*width * 4), dtype=np.float32)
    high = np.ones((height*width * 4), dtype=np.float32) * 1
    grid_low = np.zeros((4, height, width), dtype=np.float32)
    grid_high = np.ones((4, height, width), dtype=np.float32) * 1
    
    # Set the appropriate ranges for the third and fourth channels
    grid_low[2, :, :] = 0
    grid_high[2, :, :] = 1
    grid_low[3, :, :] = 0
    grid_high[3, :, :] = 1
    
    
    # Combine into a dictionary space
    observation_space = gym.spaces.Box(low=grid_low, high=grid_high, dtype=np.float32)
    
    return observation_space

class TopOptEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, height, width, bounded=[(0, 0), (-1, 0)], loaded =[(-1, -1, "LY20")], mode="train", threshold=0.67, render_mode="human", stress_constrain = 1):
        self.mode = mode
        self.threshold = threshold
        self.height = height
        self.width = width

        if self.mode=="train":
            self.loaded = self.get_random_loaded(self.height, self.width)
            self.bounded = self.get_random_bounded(self.height, self.width,self.loaded)
        else:
            self.bounded = bounded
            self.loaded = loaded

        self.grid = dsf.create_grid(self.height, self.width, self.bounded, self.loaded)
        vm_stresses, self.init_avg_strain = get_needed_fem_values(self.grid)
        self.grid = dsf.convert_grid_with_von_mises(self.grid, vm_stresses)
        self.action_space = gym.spaces.Discrete(height * width)
        self.observation_space = get_observation_space(self.height, self.width)

        self.constraint = 0.0
        self.step_count = 0
        self.reward = 0
        self.accumulated_reward = 0

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = False
        #print(self.grid)
        if self.is_illegal_action(action, verbose=True):
            self.reward = -1
            terminated = True 
            obs = self.create_observation()
        else:
            self.grid = self.take_action(action)

            self.reward, vm_stresses = get_reward(self.grid, self.init_avg_strain, plot=False)
            self.grid = dsf.convert_grid_with_von_mises(self.grid, vm_stresses)
            if self.get_constraint(self.grid) >= 1 - self.threshold:
                # print(self.get_constraint(self.grid))
                # print("Threshold reached")
                self.reward+= 3
                terminated = True
                truncated = True
                obs = self.create_observation()
        #print(self.grid)
        self.accumulated_reward += self.reward
        #print("Reward: ", self.reward)
        obs = self.create_observation()
        return obs, self.reward, terminated, truncated, self.get_info()
    
    def create_observation(self):
        obs = self.grid.copy()


        # print("Grid: \n", grid[0,:,:].round(2))

        stresses = obs[0,:,:]
        stresses = (stresses * 255).astype(np.uint8)
        #stresses[stresses > 255] -= 1       
        stresses[self.grid[0, :, :] == 0] = 0
        # print("Stresses: \n", stresses)

        for row in range(self.height):
            for col in range(self.width):
                if stresses[row, col] == 0 and self.grid[0, row, col] != 0:
                    stresses[row, col] = 1


        bounded = obs[1,:,:]
        bounded[bounded != 0] = 255
        #bounded = bounded.astype(np.uint8)

        loaded = obs[2:, :, :]

        loaded[loaded != 0] += 81 + 94
        #loaded = loaded.astype(np.uint8)
        obs[0,:,:] = stresses
        obs[1,:,:] = bounded
        obs[2:,:,:] = loaded

        return obs/255
    def reset(self, seed=None, options=None):
        if self.mode == "train":
            self.loaded = self.get_random_loaded(self.height, self.width)
            self.bounded = self.get_random_bounded(self.height, self.width, self.loaded)
            self.grid = dsf.create_grid(self.height, self.width, self.bounded, self.loaded)
            vm_stresses, self.init_avg_strain = get_needed_fem_values(self.grid)
            self.grid = dsf.convert_grid_with_von_mises(self.grid, vm_stresses)
        else:
            self.loaded = self.loaded
            self.bounded = self.bounded
            self.grid = dsf.create_grid(self.height, self.width, self.bounded, self.loaded)
            vm_stresses, self.init_avg_strain = get_needed_fem_values(self.grid)
            self.grid = dsf.convert_grid_with_von_mises(self.grid, vm_stresses)
        


        self.step_count = 0
        self.reward = 0
        obs = self.create_observation() 

        return obs, self.get_info()

    def print(self, mode="human"):
        if mode == "human":
            #print(self.grid)
            a, b, c, d = dsf.extract_fem_data(self.grid)
            fem.plot_mesh(a, b)
            fem.FEM(a, b, c, d, plot_flag=True, grid=self.grid)
        elif mode == "rgb_array":
            raise NotImplementedError

    def is_train(self):
        return self.mode == "train"
    def is_eval(self):
        return self.mode == "eval"
    
    def are_all_bounded_nodes_connected_to_all_loaded_nodes(self, row, col):
        # Temporarily remove the node
        original_value = self.grid[0, row, col]
        self.grid[0, row, col] = 0

        # Perform DFS or BFS to check connectivity
        visited = np.zeros((self.height, self.width), dtype=bool)
        stack = []

        # Find all bounded nodes
        bounded_nodes = [(r, c) for r in range(self.height) for c in range(self.width) if self.grid[1, r, c] == 1]

        # Find all loaded nodes
        loaded_nodes = [(r, c) for r in range(self.height) for c in range(self.width) if self.grid[2, r, c] != 0 or self.grid[3, r, c] != 0]

        if not bounded_nodes or not loaded_nodes:
            self.grid[0, row, col] = original_value  # Restore the node
            return False

        # Perform DFS from the first bounded node
        start_node = bounded_nodes[0]
        stack.append(start_node)
        while stack:
            r, c = stack.pop()
            if visited[r, c]:
                continue
            visited[r, c] = True
            # Add neighbors to the stack
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and not visited[nr, nc] and self.grid[0, nr, nc] != 0:
                    stack.append((nr, nc))

        # Check if all bounded nodes and loaded nodes are visited
        all_connected = all(visited[r, c] for r, c in bounded_nodes) and all(visited[r, c] for r, c in loaded_nodes)

        self.grid[0, row, col] = original_value  # Restore the node
        return all_connected

    def is_illegal_action(self, action, verbose=False):
        row, col = self.action_to_coordinates(action)
        if self.grid[0, row, col] == 0:
            # if verbose:
            #     print("Tried removing material from an empty cell.")
            return True
        if self.grid[1, row, col] != 0:
            # if verbose:
            #     print("Tried removing material from a bounded cell.")
            return True
        if self.grid[2, row, col] != 0 or self.grid[3, row, col] != 0:
            # if verbose:
                #print("Tried removing material from a loaded cell.")
            return True
        if not self.are_all_bounded_nodes_connected_to_all_loaded_nodes(row, col):
            # if verbose:
            #     print("Action would disconnect bounded nodes from loaded nodes.")
            return True
        return False

    def get_constraint(self, grid):
        self.constraint = (self.height* self.width - (np.count_nonzero(grid[0, :, :]))) / (self.height * self.width)
        #print ("Constraint: ", self.constraint)
        return self.constraint
    def take_action(self, action):
        row, col = self.action_to_coordinates(action)
        return dsf.remove_material(self.grid, row, col)

    def action_to_coordinates(self, action):
        row, col = divmod(action, self.width)
        return row, col

    def get_random_loaded(self, height, width):
        loaded = []
        row = self.get_random_number(0, height - 1)
        col = self.get_random_number(0, width - 1)
        negative_bool = self.get_random_number(0, 1)
        x_or_y = self.get_random_number(0, 1)
        if negative_bool == 0:
            load_value = self.get_random_number(10, 80)
        else:
            load_value = -self.get_random_number(10, 80)
        if x_or_y == 0:
            loaded.append((row, col, "LX" + str(load_value)))
        else:
            loaded.append((row, col, "LY" + str(load_value)))
        return loaded
    
    def get_random_number(self, low, high):
        return random.randint(low, high)

    def get_random_edge_coordinate(self, height, width):
        edge = self.get_random_number(0, 3)
        if edge == 0:  # Top edge
            row = 0
            col = self.get_random_number(0, width - 1)
        elif edge == 1:  # Bottom edge
            row = height - 1
            col = self.get_random_number(0, width - 1)
        elif edge == 2:  # Left edge
            row = self.get_random_number(0, height - 1)
            col = 0
        else:  # Right edge
            row = self.get_random_number(0, height - 1)
            col = width - 1
        return row, col

    def get_random_bounded(self, height, width, loaded):
        bounded = []
        num_bounded = self.get_random_number(1, max(height, width))
        for _ in range(num_bounded):
            while True:
                row, col = self.get_random_edge_coordinate(height, width)
                if (row, col) in bounded:
                    continue
                # Check if the node is adjacent to any loaded node
                is_adjacent_to_loaded = any(
                    (row == lr and col == lc) or
                    (row + dr == lr and col + dc == lc)
                    for lr, lc, _ in loaded
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                )
                if is_adjacent_to_loaded:
                    continue
                bounded.append((row, col))
                break
        return bounded
    
    def get_info(self):
        return {"grid" : self.grid, 
                "reward" : self.reward, 
                "step" : self.step_count}

def make_env(height, width, bounded, loaded, mode="train", threshold=0.67):
    def _init():
        env = TopOptEnv(height, width, bounded, loaded, mode, threshold)
        return env
    return _init


def poly_matrix(x, y, order=2):
    """ Function to produce a matrix built on a quadratic surface """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G   


def Reward_Surface(opts):
    x=np.array([1,0,0,1,.5,0,.5])
    y=np.array([0,0,1,1,.5,.5,0])
    z=np.array([])
    a=5
    b=5
    for i in range(0,len(x)):
        z=np.append(z,(a*(x[i])**2)+(b*(y[i])**2))
    
    ordr=2
    G = poly_matrix(x, y, ordr)
    # Solve for np.dot(G, m) = z:
    m = np.linalg.lstsq(G, z,rcond=None)[0]
    nx, ny = 1000, 1000
    
    xx, yy = np.meshgrid(np.linspace(0, 1, nx),
                         np.linspace(0, 1, ny))
    GoG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    zz = np.reshape(np.dot(GoG, m), xx.shape)
    this_dir, this_filename = os.path.split(__file__)
    base_folder =this_dir
    with open(base_folder+'/Trial_Data/Reward_Data.npy','rb') as f:
        Data = np.load(f)
    X_Data=Data[:,0]
    Y_Data=Data[:,1]
    Z_Data=Data[:,2]

    GG = poly_matrix(X_Data, Y_Data, ordr)
# Solve for np.dot(G, m) = z:
    mm = np.linalg.lstsq(GG, Z_Data,rcond=None)[0]

    GoGG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    Reward_Ind = np.reshape(np.dot(GoGG, mm), xx.shape)[:,-1]
    return zz,Reward_Ind



def main():
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0),(-1, 0)]
    loaded = [(-1, -1, "LY20")]

    # Create the environment
    env = TopOptEnv(height, width, bounded, loaded, mode="eval")
    obs, info = env.reset()
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        #print("\n----------------------------------------------------------")
        #print("before step:", _)
        env.print()
        #print("GRID \n", env.grid[0,:,:].round(3))
        print("OBS:\n", obs[0,:,:].round(3))
        #action = env.action_space.sample()
        action = take_best_action(obs)
        #action = take_worst_action(obs)
        print( "Action: ", action)
        #print("Action: ", env.action_to_coordinates(action))
        #print( "Action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Reward: ", reward)
        #print("\nGRID after step \n", env.grid[0,:,:].round(3))
        #print("OBS after steo:\n", obs[0,:,:].round(3))
        #env.print()
        if terminated:
            obs, info = env.reset()
            break

def take_best_action(obs):
    y = obs.shape[2]  # Get the width of the 2D matrix within the 3D observation array
    i, j = np.unravel_index(np.argmax(obs[0, :, :]), obs[0, :, :].shape)  # Find the coordinates of the maximum value
    action = i * y + j  # Translate the coordinates to an action
    return action

def take_worst_action(obs):
    y = obs.shape[2]  # Get the width of the 2D matrix within the 3D observation array

    # Create a masked array where values <= 1 are masked
    masked_obs = np.ma.masked_less_equal(obs[0, :, :], 0.005)

    # Find the index of the minimum value that is larger than 1
    min_index = np.argmin(masked_obs)

    # Translate the index to 2D coordinates
    i, j = np.unravel_index(min_index, obs[0, :, :].shape)

    # Translate the coordinates to an action
    action = i * y + j
    return action

    
if __name__ == "__main__":
    main()

