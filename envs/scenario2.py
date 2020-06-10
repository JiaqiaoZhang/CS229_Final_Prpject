import gym
from gym.spaces import Box, Discrete
from gym.utils import seeding
import numpy as np

from .world import World
from .agents import Car, Building, Pedestrian, Painting
from .geometry import Point
import time

class Scenario2(gym.Env):
	def __init__(self):
		self.seed(0) # just in case we forget seeding
		
		self.init_ego = Car(Point(22, 10), heading = np.pi/2)
		self.init_ego.velocity = Point(0, 10.)
		self.init_adv = Car(Point(22, 40), heading = np.pi/2, color='blue')
		self.init_adv.velocity = Point(0, 8.)
		
		self.slowdown_point = Point(22, 80)
		self.stop_duration = 3.
		self.target = Point(22, 120)
		
		self.noise_adv_pos = 1.0
		self.noise_adv_vel = 1.0
		self.dt = 0.1
		self.T = 40
		
		self.initiate_world()
		self.reset()
		
	def initiate_world(self):
		self.world = World(self.dt, width = 40, height = 120, ppm = 5)
		self.world.add(Building(Point(8, 60), Point(16, 120)))
		self.world.add(Building(Point(32, 60), Point(16, 120)))

	def reset(self):
		self.ego = self.init_ego.copy()
		self.ego.min_speed = 0.
		self.ego.max_speed = 20.
		self.adv = self.init_adv.copy()
		self.adv.min_speed = 0.
		self.adv.max_speed = 10.
		self.aggressive_safe_distance = 15.
		self.add_noise()
		
		self.slowdown_t = np.inf

		self.world.reset()

		self.world.add(self.ego)
		self.world.add(self.adv)
		
		return self._get_obs()
		
	def close(self):
		self.world.close()
		
	def add_noise(self):
		self.ego.center += Point(0, 20*self.np_random.rand() - 10)
		self.adv.center += Point(0, 20*self.np_random.rand() - 10)
		self.aggressive_safe_distance += self.np_random.rand()*4 - 2

	@property
	def observation_space(self):
		low = np.array([0, self.ego.min_speed, 30, self.adv.min_speed - self.noise_adv_vel/2.])
		high= np.array([self.target.y + self.ego.max_speed*self.dt, self.ego.max_speed, self.target.y + self.adv.max_speed*self.dt, self.adv.max_speed + self.noise_adv_vel/2.])
		return Box(low=low, high=high)

	@property
	def action_space(self):
		return Box(low=np.array([-3.5]), high=np.array([2.]))
	
	def seed(self, seed):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def get_adv_control(self):
		if self.adv.y < self.slowdown_point.y:
			if self.adv.speed > self.init_adv.speed:
				return np.array([0, 0.], dtype=np.float32)
			else:
				return np.array([0, 1. + self.np_random.rand()*0.4 - 0.2], dtype=np.float32)
		elif self.world.t < self.slowdown_t + self.stop_duration: # the adversarial car has just passed the slowdown point
			if self.slowdown_t > self.T:
				self.slowdown_t = self.world.t
			return np.array([0, -3 + self.np_random.rand()*0.4 - 0.2], dtype=np.float32)
		else:
			return np.array([0, 2. + self.np_random.rand()*0.4 - 0.2], dtype=np.float32)
		
	def get_ego_control(self,policy_no=0):
		if policy_no==0: # aggressive
			if self.adv.y - self.ego.y > np.maximum(np.minimum(self.aggressive_safe_distance, 2*self.ego.speed), 1):
				return np.array([0, 1.5 + self.np_random.rand()*0.4 - 0.2], dtype=np.float32)
			elif self.ego.speed < 2.:
				return np.array([0, 0.], dtype=np.float32)
			else:
				return np.array([0, -3.4 + self.np_random.rand()*0.2 - 0.1], dtype=np.float32)
		
		elif policy_no==1: # cautious
			if self.adv.y - self.ego.y > np.maximum(2*self.ego.speed, 1):
				return np.array([0, 0.5 + self.np_random.rand()*0.4 - 0.2], dtype=np.float32)
			elif self.ego.speed < 2.:
				return np.array([0, 0.], dtype=np.float32)
			else:
				return np.array([0, -2.5 + self.np_random.rand()*0.4 - 0.2], dtype=np.float32)

		# elif policy_no == 2:  # normal
		# 	ttw_ego = (self.wall.y - self.ego.y) / np.abs(self.ego.yp + 1e-8)
		# 	if ttc_ego < 0.05 or ttc_adv < 0:
		# 		return np.array([0, 1.95 + 0.05 * self.np_random.rand()], dtype=np.float32)
		# 	elif ttw_ego > 1.0 and ttw_ego < 3:
		# 		return np.array([0, 0], dtype=np.float32)
		# 	elif ttc_ego < ttc_adv - 0.2 or not self.ego_can_see_adv:
		# 		return np.array([0, np.minimum(1.5, np.maximum(1, self.ego.inputAcceleration + self.np_random.rand() * 0.2 - 0.1))],dtype=np.float32)
		# 	else:
		# 		return np.array([0, -3 - np.random.rand() * 0.25], dtype=np.float32)
	@property
	def target_reached(self):
		return self.ego.y >= self.target.y
	
	@property
	def collision_exists(self):
		return self.ego.collidesWith(self.adv)
		
	def step(self, action):
		while type(action) == list:
			action = action[0]
		action = np.clip(action, self.action_space.low, self.action_space.high)
		
		ego_action = np.array([0, action], dtype=np.float32)
		adv_action = self.get_adv_control()
		
		self.ego.set_control(*ego_action)
		self.adv.set_control(*adv_action)
		
		self.world.tick()
		
		return self._get_obs(), 0, self.collision_exists or self.target_reached or self.world.t >= self.T, self.world.t
		
	def _get_obs(self):
		return np.array([self.ego.center.y, self.ego.velocity.y, self.adv.center.y + self.noise_adv_pos*self.np_random.rand() - self.noise_adv_pos/2., self.adv.velocity.y + self.noise_adv_vel*self.np_random.rand() - self.noise_adv_vel/2.])
		

	def render(self, mode='rgb'):
		self.world.render()