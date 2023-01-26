from enum import auto
import pygame
import math
import numpy as np
import gym
from gym import spaces
import time
import random

class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, auto_render=False, auto_sleep=0):
        super(GameEnv, self).__init__()
        self.auto_render = auto_render
        self.auto_sleep = auto_sleep
        self.pygame_inited = False
        obs = self.reset()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(obs),), dtype="float32")

    def step(self, action):
        print(f"PRE: {self.ball_pos}")
        if self.auto_render:
            self.render()
        time.sleep(self.auto_sleep)

        self.iter += 1
        if action == 1:
            self.bar_pos[0] = min(self.bar_pos[0] + self.bar_dist_per_move, self.width - self.bar_width)
        elif action == 2:
            self.bar_pos[0] = max(self.bar_pos[0] - self.bar_dist_per_move, 0)
        else:
            assert action == 0

        # Check ball collision
        if self.ball_top_bar_collision():
            if not self.colliding:
                whole_range = (1 + self.tolerance_factor) * (self.bar_width + self.ball_size)
                perc = (self.ball_pos[0] + self.ball_size - self.bar_pos[0]) / whole_range
                self.ball_dir = math.pi * perc
            self.colliding = True
        else:
            self.colliding = False
        if self.ball_left_wall_collision() or self.ball_right_wall_collision():
            self.ball_dir = math.pi - self.ball_dir
        self.do_ball_box_collision(self.ball_dir)
        self.ball_dir = self.ball_dir % (2 * math.pi)

        if self.iter % self.ball_move_freq == 0:
            self.ball_pos[0] = self.ball_pos[0] - self.ball_dist_per_move * math.cos(self.ball_dir)
            self.ball_pos[1] = self.ball_pos[1] - self.ball_dist_per_move * math.sin(self.ball_dir)
        game_lost = self.ball_pos[1] > self.bar_pos[1] + 1
        print(f"POST-: {self.ball_pos}")

        if game_lost:
            return self.reset(), -1000, True, dict()

        observations = self.compute_obervations()
        reward = np.sum(self.boxes.flatten() == 0)
        done = reward == len(self.boxes) * len(self.boxes[0])
        return observations, reward, done, dict()

    def compute_obervations(self):
        # observations = list(self.boxes.flatten())
        observations = []
        observations.append(self.bar_pos[0])
        observations.append(self.bar_pos[1])
        observations.append(self.ball_pos[0])
        observations.append(self.ball_pos[1])
        observations.append(self.ball_dir)
        return np.array(observations)

    def reset(self):
        self.width = 700
        self.height = 700
        self.block_size = 25
        assert self.width % self.block_size == 0
        assert self.height % self.block_size == 0
        self.show_size = self.block_size - 2
        self.bar_width = 200
        self.bar_height = 5
        self.ball_size = 10

        self.game_over = False
        self.boxes = self.prepare_boxes()
        self.bar_pos = [(self.width - self.bar_width) // 2, self.height - 50]
        self.ball_pos = [self.bar_pos[0]-10, self.bar_pos[1]-10]
        self.ball_dir = random.random() * math.pi
        self.ball_move_freq = 1
        self.ball_dist_per_move = 15
        self.iter = 0
        self.bar_dist_per_move = 10
        self.colliding = False
        self.tolerance_factor = 0.5

        return self.compute_obervations()

    def render(self, mode='human'):
        if not self.pygame_inited:
            self.pygame_inited = True
            self.red = (255, 0, 0)
            self.white = (255, 255, 255)
            self.black = (0, 0, 0)
            print("init pygame...")
            pygame.init()
            self.dis = pygame.display.set_mode((self.width, self.height))
        self.dis.fill(self.black)
        self.show_boxes()
        pygame.draw.rect(self.dis, self.white, [self.bar_pos[0], self.bar_pos[1], self.bar_width, self.bar_height])
        pygame.draw.rect(self.dis, self.red, [self.ball_pos[0], self.ball_pos[1], self.ball_size, self.ball_size])
        pygame.display.update()

    def close (self):
        pass

    @classmethod
    def human_play(cls, clock_rate):
        env = GameEnv(auto_render=False)
        env.reset()
        env.render()
        game_over = False
        clock = pygame.time.Clock()
        while True:
            pygame.event.get()
            if game_over:
                game_over = False
                env.reset()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                env.step(2)
            elif keys[pygame.K_RIGHT]:
                env.step(1)
            else:
                env.step(0)
            env.render()
            clock.tick(clock_rate)

    def show_boxes(self):
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes[0])):
                if self.boxes[i][j]:
                    pygame.draw.rect(self.dis, self.white, [i*self.block_size, j*self.block_size, self.show_size, self.show_size])

    def prepare_boxes(self):
        return np.ones((self.width // self.block_size, 3))

    def ball_top_bar_collision(self):
        return (
            self.ball_pos[0] + self.ball_size > self.bar_pos[0] and
            self.ball_pos[0] < self.bar_pos[0] + self.bar_width and
            self.ball_pos[1] + self.ball_size > self.bar_pos[1] and
            self.ball_pos[1] < self.bar_pos[1] + self.bar_height
        )

    def ball_left_wall_collision(self):
        return self.ball_pos[0] < 0

    def ball_right_wall_collision(self):
        return self.ball_pos[0] + self.ball_size > self.width

    def do_ball_box_collision(self, og_dir):
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes[i])):
                if self.boxes[i][j]:
                    box_x = i*self.block_size
                    box_y = j*self.block_size
                    if not -og_dir == self.ball_dir and \
                        self.ball_pos[0] + self.ball_size > box_x and \
                        self.ball_pos[0] < box_x + self.block_size and \
                        self.ball_pos[1] + self.ball_size > box_y and \
                        self.ball_pos[1] < self.block_size + box_y:
                            self.ball_dir = -self.ball_dir
                            self.boxes[i][j] = 0

def demo():
    env = GameEnv()
    import random
    import time
    while True:
        action = random.randint(0,2)
        _, reward, _, _ = env.step(action)
        env.render()
        print(reward)
        time.sleep(0.02)

def train():
    from stable_baselines3 import PPO
    env = GameEnv(auto_render=False, auto_sleep=0)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=10**4)
    model.learn(total_timesteps=10**6)
    model.save("output_model")
    return model

def test(model):
    print("-"*50)
    print("Starting test...")
    env = GameEnv(auto_render=True, auto_sleep=0.015)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        env.step(action)

# model = train()
# test(model)
GameEnv.human_play(30)
