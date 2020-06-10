import numpy as np
import os
from envs.geometry import Point

scenario_names = ['data01','data01_new', 'data23', 'data45']
obs_sizes = {'data01':4, 'data01_new':4, 'data23':4, 'data45':5}
goals = {'data01': ['aggressive', 'timid'],'data01_new': ['aggressive', 'timid', 'normal'], 'data23': ['aggressive', 'timid', 'mix'], 'data45': ['aggressive', 'timid', 'mix']}
steering_lims = {'intersection': [-0.5,0.5], 'circularroad': [-0.15,0.15], 'lanechange': [-0.15, 0.15]}

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try: 
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def load_data(args):
    goal_name = args.goal.lower()
    scenario_name = args.scenario.lower()
      
    assert scenario_name in goals.keys(), '--scenario argument is invalid!'
    data = {}
    if goal_name == 'all':
        np_data = np.load('data/' + scenario_name + '.npy')
        data['u_train'] = np_data[:200000, -1]
        # np_data = [np.load('data/' + scenario_name + '_' + dn + '.npy') for dn in goals[scenario_name]]
        # print(('data/' + scenario_name + '_' + dn + '.npy') for dn in goals[scenario_name])
        # u = np.vstack([np.ones((np_data[i].shape[0],1))*i for i in range(len(np_data))])
        # np_data = np.vstack(np_data)
        # data['u_train'] = np.array(u).astype('uint8').reshape(-1,1)
    else:
        assert goal_name in goals[scenario_name], '--data argument is invalid!'
        np_data = np.load('data/' + scenario_name + '.npy')
    goal = 0
    data['type'] = np_data[:, -1]
    if goal_name == 'aggressive':
        goal = 0
    elif goal_name == 'timid':
        goal = 1

    if goal_name != 'all':
        np_data = np_data[data['type'] == goal]
      #0: aggressive, 1: timid
    data['x_train'] = np_data[:200000,:5].astype('float32')
    data['y_train'] = np_data[:200000,-2].astype('float32').reshape((-1, 1)) # control is always 2D: throttle and steering
    return data
    
   
def optimal_act_circularroad(env, d):
    if env.ego.speed > 10:
        throttle = 0.06 + np.random.randn()*0.02
    else:
        throttle = 0.6 + np.random.randn()*0.1
        
    # setting the steering is not fun. Let's practice some trigonometry
    r1 = 30.0 # inner building radius (not used rn)
    r2 = 39.2 # inner ring radius
    R = 32.3 # desired radius
    if d==1: R += 4.9
    Rp = np.sqrt(r2**2 - R**2) # distance between current "target" point and the current desired point
    theta = np.arctan2(env.ego.y - 60, env.ego.x - 60)
    target = Point(60 + R*np.cos(theta) + Rp*np.cos(3*np.pi/2-theta), 60 + R*np.sin(theta) - Rp*np.sin(3*np.pi/2-theta)) # this is pure magic (or I need to draw it to explain)
    desired_heading = np.arctan2(target.y - env.ego.y, target.x - env.ego.x) % (2*np.pi)
    h = np.array([env.ego.heading, env.ego.heading - 2*np.pi])
    hi = np.argmin(np.abs(desired_heading - h))
    if desired_heading >= h[hi]: steering = 0.15 + np.random.randn()*0.05
    else: steering = -0.15 + np.random.randn()*0.05
    return np.array([steering, throttle]).reshape(1,-1)
    
    
def optimal_act_lanechange(env, d):
    if env.ego.speed > 10:
        throttle = 0.06 + np.random.randn()*0.02
    else:
        throttle = 0.8 + np.random.randn()*0.1
        
    if d==0:
        target = Point(37.55, env.ego.y + env.ego.speed*3)
    elif d==1:
        target = Point(42.45, env.ego.y + env.ego.speed*3)
    desired_heading = np.arctan2(target.y - env.ego.y, target.x - env.ego.x) % (2*np.pi)
    h = np.array([env.ego.heading, env.ego.heading - 2*np.pi])
    hi = np.argmin(np.abs(desired_heading - h))
    if desired_heading >= h[hi]: steering = 0.15 + np.random.randn()*0.05
    else: steering = -0.15 + np.random.randn()*0.05
    return np.array([steering, throttle]).reshape(1,-1)