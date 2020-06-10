import numpy as np
from train_coil_pj import NN
# import gym_carlo
import gym
import time
import argparse
import tensorflow as tf
# from gym_carlo.envs.interactive_controllers import GoalController
# from gym_carlo.envs.interactive_controllers import Data01Controller
from utils import *
from envs.scenario0 import *
from envs.scenario4 import *
def controller_mapping(scenario_name, control):
    """Different scenarios have different number of goals, so let's just clip the user input -- also could be done via np.clip"""
    if control >= len(goals[scenario_name]):
        control = len(goals[scenario_name])-1
    return control

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="data01, data23, data45", default="data01")
    parser.add_argument('--goal', type=str, help="aggressive, timid, mix, normal", default="all")
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, '--scenario argument is invalid!'
    
    # if args.goal.lower() == 'all':
    #     goal_id = len(goals[scenario_name])
    # else:
    #     goal_id = np.argwhere(np.array(goals[scenario_name])==args.goal.lower())[0,0] # hmm, unreadable

    goal_id = -1
    if args.goal.lower() == 'aggressive':
        goal_id = 0
    elif args.goal.lower() == 'timid':
        goal_id = 1
    elif args.goal.lower() == 'normal':
        goal_id = 2
    elif args.goal.lower() == 'mix':
        goal_id = 3

    # env = gym.make(scenario_name + 'Scenario-v0', goal=goal_id)
    env = Scenario4()

    nn_model = NN(obs_sizes[scenario_name],1)
    nn_model.load_weights('./policies/' + scenario_name + '_all_CoIL')

    episode_number = 10 if args.visualize else 100
    success_counter = 0
    time_counter = 0
    env.T = 400*env.dt - env.dt/2. # Run for at most 200dt = 40 seconds
    for _ in range(episode_number):
        env.seed(int(np.random.rand()*1e6))
        obs, done = env.reset(), False
        if args.visualize:
            env.render()
            # interactive_policy = Data01Controller(env.world)
        while not done:
            t = time.time()
            obs = np.array(obs).reshape(1,-1)
            # u = controller_mapping(scenario_name, interactive_policy.control) if args.visualize else goal_id

            # u = interactive_policy.control if args.visualize else goal_id

            # if args.visualize:
            #     # u = interactive_policy.control
            if goal_id == 3:
                u = np.random.randint(2, size=1)
            else:
                u = goal_id

            print("u: ", u)

            action = nn_model(obs,u).numpy().reshape(-1)

            # print("*****")
            print("action", action)

            obs,_,done, terminal_time = env.step(np.array(action))

            if args.visualize: 
                env.render()
                while time.time() - t < env.dt/2: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                if args.visualize:
                    time.sleep(1)
                    env.close()
                if env.target_reached:
                    success_counter += 1
                    time_counter += terminal_time
    if not args.visualize:
        print('Success Rate = ' + str(float(success_counter)/episode_number))
        print('Mean reach time = ' + str(float(time_counter / success_counter)))
