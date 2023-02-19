TRAIN = False

#ENV_NAME = 'RiverraidDeterministic-v4'
#ENV_NAME = 'AtlantisDeterministic-v4'
#ENV_NAME = 'SpaceInvaders-v4'
ENV_NAME = 'BreakoutDeterministic-v4'
#ENV_NAME = 'PongDeterministic-v4'  

import os
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import pickle
from Atari import Atari
from DqnLeaky import DQN
from ReplayMemory import ReplayMemory
from ExplorationExploatation import ExplorationExploitationScheduler
from TargetNetworkUpdater import TargetNetworkUpdater
from utils import learn, clip_reward, generate_gif

tf.disable_v2_behavior()

tf.reset_default_graph()

# hiperparametrii
MAX_EPISODE_LENGTH = 18000       # lungimea maxima a unui episod
EVAL_FREQUENCY = 200000          # frecventa la care evaluam agentul
EVAL_STEPS = 18000               # numarul de pasi intr-o evalauare
NETW_UPDATE_FREQ = 10000         # numarul de pasi la care actualizam reteaua tinta cu cea principala

DISCOUNT_FACTOR = 0.99           # gamma
REPLAY_MEMORY_START_SIZE = 50000 # dimensiunea minima a meoriei de replauy 
                                 
MAX_FRAMES = 50000000            # Numarul total de pasi in invatare 
MEMORY_SIZE = 1000000            # dimesniunea memoriei de replay
NO_OP_STEPS = 10                 # numarul de pasi de NOOP sau tragere la inceputul unui joc
                                 
UPDATE_FREQ = 4                  # la fiecare patru pasi, calculam gradientul negativ
HIDDEN = 1024                    # numarul de neuroni in ultimul strat ascuns
LEARNING_RATE = 0.00001          # rata de invatare 

BS = 32                          # dimesniunea minilotului

PATH = "output/"                 # salvam gif-urile si modelele
SUMMARIES = "summaries"          # logdir pentru tensorboard
RUNID = 'run_8'                  # numele sesiunii de tensorboard
os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n, atari.env.unwrapped.get_action_meanings()))

# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE) 
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN) 

init = tf.global_variables_initializer()
saver = tf.train.Saver()    

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

LAYER_IDS = ["conv1", "conv2", "conv3", "flat", "denseAdvantage", "denseAdvantageBias", "denseValue", "denseValueBias"]

# Scalarii pentru tensorboard: loss, average reward si evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

# Histogramele pentru tensorboard: parameters
with tf.name_scope('Parameters'):
    ALL_PARAM_SUMMARIES = []
    for i, Id in enumerate(LAYER_IDS):
        with tf.name_scope('mainDQN/'):
            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)

def train():
    """training si evaluare"""
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS) 
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    
    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n,  epsilon_initial=0.01, 
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
        max_frames=MAX_FRAMES)

    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, "output/my_model-20269532") # activeaza daca reiei un model
        # update_networks(sess) # activeaza daca reiei un model
         
        frame_number = 0 # daca reiei un model, pui numarul pasului, altfel 0
        rewards = []
        loss_list = []

        # infile = open('replay_mem', 'rb') # comentezi daca nu incarci model
        # my_replay_memory = pickle.load(infile) # comentezi daca nu incarci model
        # infile.close() # comentezi daca nu incarci model

        # infile = open('loss', 'rb') # comentezi daca nu incarci model
        # loss_list = pickle.load(infile) # comentezi daca nu incarci model
        # infile.close() # comentezi daca nu incarci model

        
        while frame_number < MAX_FRAMES:
            
            ########################
            ####### Training #######
            ########################
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                done_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    action = explore_exploit_sched.get_action(sess, frame_number, atari.state)   
                    processed_new_frame, reward, done, done_life_lost, _ = atari.step(sess, action)  
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    clipped_reward = clip_reward(reward)

                    my_replay_memory.add_experience(action=action, frame=processed_new_frame[:, :, 0],reward=clipped_reward, done=done_life_lost)   
                    
                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,BS, gamma = DISCOUNT_FACTOR)
                        loss_list.append(loss)
                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        update_networks(sess)
                    
                    if done:
                        done = False
                        break

                rewards.append(episode_reward_sum)
                
                if len(rewards) % 10 == 0:
                    if frame_number > REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(PERFORMANCE_SUMMARIES, feed_dict={LOSS_PH:np.mean(loss_list), REWARD_PH:np.mean(rewards[-100:])})
                        
                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    summ_param = sess.run(PARAM_SUMMARIES)
                    SUMM_WRITER.add_summary(summ_param, frame_number)
                    
                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number, 
                              np.mean(rewards[-100:]), file=reward_file)
            
            ########################
            ###### Evaluare ######
            ########################
            done = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0
            
            for _ in range(EVAL_STEPS):
                if done:
                    done_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    done = False
               
                # daca agentul a murit, trage, pentru ca in unele jocuri nu se intampla nimic pana nu tragi
                action = 1 if done_life_lost else explore_exploit_sched.get_action(sess, frame_number,atari.state, evaluation=True)
                
                processed_new_frame, reward, done, done_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif: 
                    frames_for_gif.append(new_frame)
                if done:
                    eval_rewards.append(episode_reward_sum)
                    gif = False # Savam doar primul joc din evalauare ca si gif
                     
            print("Evaluation score:\n", np.mean(eval_rewards))       
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                print("No evaluation game finished")
            
            #Salvam parametrii retelei
            saver.save(sess, PATH+'/my_model', global_step=frame_number)
            outfile = open('replay_mem', 'wb')
            pickle.dump(my_replay_memory, outfile)
            outfile.close()
            outfile = open('loss', 'wb')
            pickle.dump(loss_list, outfile)
            outfile.close()
            frames_for_gif = []
            graph = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID, 'graphs'), sess.graph)
            graph.close()
            
            
            # Afiseaza scorul in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)

if TRAIN:
    train()


save_files_dict = {
    'BreakoutDeterministic-v4':("results/breakout-leaky/", "my_model-27300331.meta", "my_model-27300331"),
    'PongDeterministic-v4':("trained/pong/", "my_model-3217770.meta")
}

if not TRAIN:
    
    gif_path = "GIF/"
    os.makedirs(gif_path, exist_ok=True)

    trained_path, save_file, chkpt = save_files_dict[ENV_NAME]

    explore_exploit_sched = ExplorationExploitationScheduler(MAIN_DQN, atari.env.action_space.n, replay_memory_start_size=REPLAY_MEMORY_START_SIZE, max_frames=MAX_FRAMES)
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(trained_path+save_file)
        saver.restore(sess,trained_path + chkpt)
        frames_for_gif = []
        done_life_lost = atari.reset(sess, evaluation = True)
        episode_reward_sum = 0
        while True:
            atari.env.render()
            action = 1 if done_life_lost else explore_exploit_sched.get_action(sess, 0, atari.state,  evaluation = True)
            
            processed_new_frame, reward, done, done_life_lost, new_frame = atari.step(sess, action)
            episode_reward_sum += reward
            frames_for_gif.append(new_frame)
            if done == True:
                break
        
        atari.env.close()
        print("The total reward is {}".format(episode_reward_sum))
        print("Creating gif...")
        generate_gif(0, frames_for_gif, episode_reward_sum, gif_path)
        print("Gif created, check the folder {}".format(gif_path))

