import tests as test
import cProfile
import re
import pstats
from pstats import SortKey
import time	

NUM_ENVS = 16
NUM_TIMESTEPS = 5e5
MODEL_NAME = "5x5 single Problem PPO standard cnn gamma 0.9 threshold 0.67 total strain"
def main(): 
    #test.env_compatability()
    #test.fem_analysis_func(strat="good", plot=True)
    #test.learn(NUM_ENVS, num_timesteps=NUM_TIMESTEPS, model_name=MODEL_NAME)
    #test.loading_learn(NUM_ENVS, num_timesteps=NUM_TIMESTEPS, model_name=MODEL_NAME)
    test.load(model_name=MODEL_NAME)
    #test.cnn()
    #test.test_all_grids()

if __name__ == "__main__":
    main()
    #start_time = time.time()
    # cProfile.run("main()", "rl_profiled_5x5_general")

    #end_time = time.time()

    #duration = end_time - start_time
    # print(f"Time taken: {duration:.2f} seconds")
    # p =pstats.Stats("rl_profiled_5x5_general")
    # p.strip_dirs().sort_stats(SortKey.FILENAME).reverse_order().print_stats(1000)