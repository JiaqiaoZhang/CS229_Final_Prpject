import numpy as np
# from .scenario0 import Scenario0
# from gym_carlo.envs.interactive_controllers import KeyboardController
# import time
# import gym
#
# s_0 = Scenario0()
# s_0.add_noise()
# s_0.render()
# dt = 0.1
# # p1.set_control(0, 0.2) # The pedestrian will have 0 steering and 0.2 acceleration. So it will not change its direction.
# # c1.set_control(0, 1.2)
# o, d = s_0.reset(), False
# interactive_policy = KeyboardController(s_0.world)
# while not d:
# 	a = [interactive_policy.steering, interactive_policy.throttle]
# 	o, _,d, _ = s_0.step(a)# This ticks the world for one time step (dt second)
# 	s_0.render()
# 	time.sleep(dt/2) # Let's watch it 2x
# #
# # 	if w.collision_exists(p1): # We can check if the Pedestrian is currently involved in a collision. We could also check c1 or c2.
# # 		print('Pedestrian has died, good job!')
# 	elif w.collision_exists(): # Or we can check if there is any collision at all.
# 		print('Collision exists somewhere...')


data = np.load('simulate_data/data01.npy')
print(s)