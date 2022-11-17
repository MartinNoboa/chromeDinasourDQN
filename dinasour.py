from mss import mss
from matplotlib import pyplot as plt
from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
import os 
import time
import numpy as np
import pydirectinput as pyinput
import pytesseract as pytess
import cv2

pytess.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class dinasour(Env):
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 350, 'left': 200, 'width': 600, 'height': 150}
        self.done_location = {'top': 350, 'left': 580, 'width': 250, 'height': 70}
        
    def step(self, action):
        actions = {0:'space',1:'down',2:'no_op'}
        if action != 2:
            pyinput.press(actions[action])
        gameOver, gameScreenshot = self.get_game_status()
        frame = self.get_observation()
        
        reward = 1
        info = {}
        return frame, reward, gameOver,info
        pass
    
    def reset(self):
        time.sleep(1)
        pyinput.press('space')
        return self.get_observation()
        
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        grayScale = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(grayScale, (100,83))
        compressed = np.reshape(resize, (1,83,100))
        return compressed
    
    def get_game_status(self):
        gameStrings = ['GAME','GAHE','GHME','GHHM','6AME','6HME','6HHE','6AHE']
        gameOver = False
        screenshot = np.array(self.cap.grab(self.done_location))
        string = pytess.image_to_string(screenshot)[:4]
        if string in gameStrings:
            gameOver = True
        return gameOver, screenshot
    
    def get_game_score(self):
        #add ocr to extract game score
        pass

  
dino = dinasour()
for episode in range(10): 
    obs = dino.reset()
    done = False  
    total_reward   = 0
    while not done: 
        obs, reward,  done, info =  dino.step(dino.action_space.sample())
        total_reward  += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))  
