import random
import gym
from FrameProcessor import FrameProcessor
import numpy as np

class Atari(object):
    """Wrapper pentru mediul gym"""
    def __init__(self, envName, no_op_steps=10, number_of_frames=4):
        self.env = gym.make(envName)
        self.process_frame = FrameProcessor()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.number_of_frames = number_of_frames

    def reset(self, sess, evaluation=False):
        """
            sess: sesiunea
            evaluation: evaluam sau antrenam
        resetam mediul si punem 4 frame-uri unul peste altul pentru primul state
        """
        frame = self.env.reset()
        self.last_lives = 0
        done_life_lost = True # Pus pe True, pentru a incepe jocul cu actiune de tragere la evaluare
                                  
        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1) 
        processed_frame = self.process_frame(sess, frame)  
        self.state = np.repeat(processed_frame, self.number_of_frames, axis=2)
        
        return done_life_lost

    def step(self, sess, action):
        """
        Executa un pas in mediul curent
            sess: sesiunea
            action: actiunea executata
        Executa o actiune si returneaza state-ul nou si recompensa
        """
        new_frame, reward, done, info = self.env.step(action)
            
        if info['ale.lives'] < self.last_lives:
            done_life_lost = True
        else:
            done_life_lost = done
        self.last_lives = info['ale.lives']
        
        processed_new_frame = self.process_frame(sess, new_frame) 
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2) 
        self.state = new_state
        
        return processed_new_frame, reward, done, done_life_lost, new_frame