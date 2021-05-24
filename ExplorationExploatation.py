import numpy as np

class ExplorationExploitationScheduler(object):
    """Determinam actiunea ce va fi luata conform cu o politica epsilon-greedy, scazand epsilon pe parcursul a 1000000 de pasi"""
    def __init__(self, DQN, n_actions, epsilon_initial=1, epsilon_final=0.1, epsilon_final_frame=0.01, epsilon_evaluation=0.0, epsilon_decrease_steps=1000000, replay_memory_start_size=50000, max_frames=25000000):
        """
            DQN: reteaua neuronala
            n_actions: numarul de actiuni posibile
            epsilon_initial: valoarea initiala a lui epsilon, favorizeaza explorarea
            replay_memory_start_size dimensiunea minima a memoriei de replay, pentru care putem extrage experiente
            epsilon_final: valoarea finala a lui epsilon, dupa replay_memory_start_size + epsilon_decrease_steps frames
            epsilon_final_frame: valoarea lui epsilon dupa max_frame; nu mai exploram apropae defel
            epsilon_evaluation: valoarea lui epsilon in timpul evaluarii
            epsilon_decrease_steps: numarul de pasi in care scadem valoarea lui epsilon 
            max_frames: numarul maxim de pasi pentru care antrenam agentul
        """
        self.n_actions = n_actions
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_final_frame = epsilon_final_frame
        self.epsilon_evaluation = epsilon_evaluation
        self.epsilon_decrease_steps = epsilon_decrease_steps
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames
        
        # rata de scadere a lui epsilon
        self.slope = -(self.epsilon_initial - self.epsilon_final)/self.epsilon_decrease_steps
        self.intercept = self.epsilon_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = -(self.epsilon_final - self.epsilon_final_frame)/(self.max_frames - self.epsilon_decrease_steps - self.replay_memory_start_size)
        self.intercept_2 = self.epsilon_final_frame - self.slope_2*self.max_frames
        
        self.DQN = DQN

    def get_action(self, session, frame_number, state, evaluation=False):
        """
            session: sesiunea de tensorflow
            frame_number: numarul pasului curent
            state: o secventa de 4 ecrane din joc (84, 84, 4)
            evaluation: evaluam sau antrenam agentul
        Returneaza:
            numarul actiunii pe care o va executa agentul
        """
        if evaluation:
            eps = self.epsilon_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.epsilon_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.epsilon_decrease_steps:
            eps = self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.epsilon_decrease_steps:
            eps = self.slope_2*frame_number + self.intercept_2
        
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        return session.run(self.DQN.best_action, feed_dict={self.DQN.input:[state]})[0]