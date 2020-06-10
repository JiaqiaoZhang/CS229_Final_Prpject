import gym
from gym.spaces import Box, Discrete
from gym.utils import seeding
import numpy as np

from .world import World
from .agents import Car, Building, Pedestrian, Painting
from .geometry import Point
import time

class Scenario5(gym.Env):
	def __init__(self):
		self.seed(0) # just in case we forget seeding
		
		self.init_ego = Car(Point(22, 10), heading = np.pi/2)
		self.init_ego.velocity = Point(0, 0.)
		self.init_adv = Car(Point(14, 115), heading = -np.pi/2, color='blue')
		self.init_adv.velocity = Point(0, 4.)
		
		self.target = Point(22, 120)
		
		self.noise_adv_pos = 1.0
		self.noise_adv_vel = 1.0
		self.dt = 0.1
		self.T = 40
		
		self.initiate_world()
		self.reset()
		
	def initiate_world(self):
		self.world = World(self.dt, width = 40, height = 120, ppm = 5)
		self.world.add(Building(Point(6, 60), Point(12, 120)))
		self.world.add(Building(Point(34, 60), Point(12, 120)))

	def reset(self):
		self.ego = self.init_ego.copy()
		self.ego.min_speed = 0.
		self.ego.max_speed = 10.
		self.adv = self.init_adv.copy()
		self.adv.min_speed = 0.
		self.adv.max_speed = 6.
		
		self.turning_point = Point(14, 90)
		self.collision_point = Point(16, 82.5)
		
		self.add_noise()

		self.world.reset()

		self.world.add(self.ego)
		self.world.add(self.adv)
		
		return self._get_obs()
		
		def close(self):
			self.world.close()
		
	def add_noise(self):
		self.ego.center += Point(0, 20*self.np_random.rand() - 10)
		self.adv.center += Point(0, 10*self.np_random.rand() - 5)
		self.collision_point.y += self.np_random.rand()*10 - 5
		self.turning_point.y += self.np_random.rand()*10 - 5

	@property
	def observation_space(self):
		low = np.array([0, self.ego.min_speed, self.init_adv.x - self.noise_adv_pos/2., 0 - self.ego.max_speed*self.dt - self.noise_adv_pos/2., self.adv.min_speed - self.noise_adv_vel/2.])
		high= np.array([self.target.y + self.ego.max_speed*self.dt, self.ego.max_speed, self.init_ego.x + self.init_ego.size.x + self.noise_adv_pos/2., self.target.y + self.noise_adv_pos/2., self.adv.max_speed + self.noise_adv_vel/2.])
		return Box(low=low, high=high)

	@property
	def action_space(self):
		return Box(low=np.array([-3.5]), high=np.array([2.]))
	
	def seed(self, seed):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def get_adv_control(self):
		ttc_ego = (self.collision_point.y - self.ego.y) / np.abs(self.ego.yp + 1e-8)
		ttc_adv = (self.adv.y - self.collision_point.y) / np.abs(self.adv.yp - 1e-8)
		if self.adv.y > self.turning_point.y:
			acceleration = 1. + self.np_random.rand()*0.4 - 0.2 if ttc_adv > ttc_ego else 0.
			return np.array([0, acceleration], dtype=np.float32)
		elif self.turning_point.y >= self.adv.y > self.collision_point.y:
			acceleration = 1. + self.np_random.rand()*0.4 - 0.2 if ttc_adv > ttc_ego else 0.
			steering = -0.1 if (self.collision_point.x - self.adv.x) * np.tan(self.adv.heading) > self.collision_point.y - self.adv.y else 0.1
			return np.array([steering, acceleration], dtype=np.float32)
		else:
			steering = -0.1 if (18 - self.adv.x) * np.tan(self.adv.heading) > -5 and np.mod(self.adv.heading, 2*np.pi) > 3*np.pi/2 else 0.1
			return np.array([steering, 0.], dtype=np.float32)
		
	def get_ego_control(self,policy_no=0):
		predicted_collision_point = (22 - self.ego.size.x/2. - self.adv.x) * np.tan(self.adv.heading) + self.adv.y
		predicted_ttc_ego = (predicted_collision_point - self.ego.y) / np.abs(self.ego.yp + 1e-8)
		predicted_ttc_adv = (self.adv.y - predicted_collision_point) / np.abs(self.adv.yp - 1e-8)
		if policy_no==0: # aggressive
			if predicted_ttc_ego < 0 or predicted_ttc_adv < -1.5 or predicted_ttc_ego < predicted_ttc_adv - 0.1:
				return np.array([0, 2.], dtype=np.float32)
			else:
				return np.array([0, -3.], dtype=np.float32)
		elif policy_no==1: # cautious
			if predicted_ttc_ego < 0 or predicted_ttc_adv < -1.5 or predicted_ttc_ego < predicted_ttc_adv - 0.5:
				return np.array([0, 1.], dtype=np.float32)
			else:
				return np.array([0, -2.5], dtype=np.float32)

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
		
		return self._get_obs(), 0, self.collision_exists or self.target_reached or self.world.t >= self.T, {}
		
	def _get_obs(self):
		return np.array([self.ego.center.y, self.ego.velocity.y, self.adv.center.x + self.noise_adv_pos*self.np_random.rand() - self.noise_adv_pos/2., self.adv.center.y + self.noise_adv_pos*self.np_random.rand() - self.noise_adv_pos/2., self.adv.velocity.y + self.noise_adv_vel*self.np_random.rand() - self.noise_adv_vel/2.])
		

	def render(self, mode='rgb'):
		self.world.render()