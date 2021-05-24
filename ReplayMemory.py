import random
import numpy as np

class ReplayMemory(object):
    """memoria de replay"""
    def __init__(self, size=1000000, screen_height=84, screen_width=84, number_of_frames=4, batch_size=32):
        """
            size: numarul de tranzitii
            screen_height: inaltimea unui ecran
            screen_width: latimea unui ecran
            number_of_frames: numarul de cadre intr-un state
            batch_size: numarul de experiente dintr-un mini-lot
        """
        self.size = size
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.number_of_frames = number_of_frames
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        # prealocarea memoriei
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.screen_height, self.screen_width), dtype=np.uint8)
        self.done_flags = np.empty(self.size, dtype=np.bool)
        
        # prealocarea memoriei pentru state-urile curente si viitoare din minilot
        self.states = np.empty((self.batch_size, self.number_of_frames, self.screen_height, self.screen_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.number_of_frames, self.screen_height, self.screen_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def add_experience(self, action, frame, reward, done):
        """
            action: actiunea pe care a executat-o agetul
            frame: cadrul curent (84, 84, 1) 
            reward: recompensa obtinuta
            done: daca jocul s-a terminta sau nu
        """
        if frame.shape != (self.screen_height, self.screen_width):
            raise ValueError('Wrong screen size')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.done_flags[self.current] = done
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
             
    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("Empty replay memory")
        if index < self.number_of_frames - 1:
            raise ValueError("index of frame needs to be at least 3")
        return self.frames[index-self.number_of_frames+1:index+1, ...]
    # memoria de replay ar trebui sa tina ultimele 1000000 de experiente; 
    # fiecare experienta are un state format din 4 cadre
    # deoarece cadrele sunt consecutive, nu vom tine 4 cadre pentru fiecare experienta, ci doar unul, deoarece celelalte 3 cadre din stiva
    # se gasesc la i-1, i-2 si i-3    
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.number_of_frames, self.count - 1)
                if index < self.number_of_frames:
                    continue
                if index >= self.current and index - self.number_of_frames <= self.current:
                    continue
                if self.done_flags[index - self.number_of_frames:index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        """
        returneaza un minilot de 32 de experiente
        """
        if self.count < self.number_of_frames:
            raise ValueError('Not enough states for a minilot')
        
        self._get_valid_indices()
            
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.done_flags[self.indices]