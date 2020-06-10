import gym
from PPO import PPO, Memory
from PIL import Image
import torch
import argparse
from utils import *
# from envs.scenario0 import *
from envs.scenario4 import *
from train_coil_pj import NN

def test():
    ############## Hyperparameters ##############
    # env_name = "LunarLander-v2"
    # creating environment
    # env = gym.make(env_name)
    env = Scenario4()
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    render = args.visualize
    max_timesteps = 500
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 100

    save_gif = False

    # filename = "PPO_{}.pth".format(env_name)
    test_name = args.case.lower()
    filename = './checkpoints/rl_checkpoint_' + test_name
    # directory = "./preTrained/"

    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    nn_model = NN(5, 1)
    # nn_model.load_weights('./policies/data01_new_all_CoIL')
    scenario_name = args.scenario.lower()
    nn_model.load_weights('./policies/' + scenario_name + '_all_CoIL')


    ppo.policy_old.load_state_dict(torch.load(filename))
    success_counter = 0
    time_counter = 0
    # env.T = 400 * env.dt - env.dt / 2.
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(max_timesteps):
            mode = ppo.policy_old.act(state, memory)
            state = np.array(state).reshape(1, -1)
            action = nn_model(state, mode).numpy().reshape(-1)
            state, reward, done, terminal_time = env.step(action)
            print("mode is ", mode)
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                ep_reward = reward
                if env.target_reached:
                    time_counter += terminal_time
                    success_counter +=1
                break
        print("timesteps", t)
        print("success number", success_counter)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        # ep_reward = 0
        # env.close()
    print('Success Rate = ' + str(float(success_counter) / ep))
    print('Mean reach time = ' + str(float(time_counter / success_counter)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="data01, data23, data45", default="data01")
    parser.add_argument('--case', type=str, help="test1, test2, test3, test4", default="test1")
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    test()
