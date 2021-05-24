
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
from tensorflow.python.ops.array_ops import zeros
from random import sample
import tensorflow as tf2

tf.disable_v2_behavior()

tf.reset_default_graph()

def plotFilters(conv_filter):
    print(conv_filter.shape)
    if (conv_filter.shape[2] > 3):
        fig, axes = plt.subplots(1, 4, figsize=(5,5))
        print("Axes are ", axes)
        # axes = axes.flatten()
        for i in range(conv_filter.shape[2]):
            plt.gray()
            axes[i].imshow(conv_filter[:,:,i])
            # ax.axis('off')
    else:
        plt.imshow(conv_filter)
    # plt.tight_layout()
    
    plt.show()

def plotActivations(activation):
    plt.imshow(activation[0,:,:])
    # plt.tight_layout()   
    plt.show()

class FrameProcessor(object):
    """Redimensioneaza frame-urile primite si le transorma in grayscale"""
    def __init__(self, screen_height=84, screen_width=84):
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.resize_images(self.processed, 
                                                [self.screen_height, self.screen_width], 
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def __call__(self, session, frame):
        """
        returneaza imaginea procesata
        """
        return session.run(self.processed, feed_dict={self.frame:frame})

class DQN(object):
    """Implementarea retelei"""
    
    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001, 
                 screen_height=84, screen_width=84, number_of_frames=4):
        """
            n_actions: numarul de actiuni, venit din mediul OpenAI Gym
            hidden: numarul de neuroni din primul strat complet conectat
            learning_rate: rata de invatare
            screen_height: inaltimea ecranului
            screen_width: latimea ecranului
            number_of_frames: numarul de ecrane cuprinse intr-o experienta
        """
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.number_of_frames = number_of_frames
        
        self.input = tf.placeholder(shape=[None, self.screen_height, self.screen_width, self.number_of_frames], dtype=tf.float32)
        # transformam intensitatile de culoare intre valori cuprinse intre 0 si 1
        self.inputscaled = self.input/255
        
        # straturile convulationale
        self.conv1 = tf.layers.conv2d(inputs=self.inputscaled, filters=32, kernel_size=[8, 8], strides=4, kernel_initializer=tf.variance_scaling_initializer(scale=2), padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2, kernel_initializer=tf.variance_scaling_initializer(scale=2),padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1, kernel_initializer=tf.variance_scaling_initializer(scale=2),padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
        self.conv4 = tf.layers.conv2d(inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1, kernel_initializer=tf.variance_scaling_initializer(scale=2),padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
        
        # impartim iesirea ultimului strat convulational intre fluxul de valoare si cel de avantaj
        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)
        self.advantage = tf.layers.dense(inputs=self.advantagestream, units=self.n_actions, kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")
        self.value = tf.layers.dense(inputs=self.valuestream, units=1, kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')
        
        # calculam valoarea functiei Q
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.best_action = tf.argmax(self.q_values, 1)
        
        # placeholderi
        # Q = r + gamma*max Q', calculat in learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # actiunea executata
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # valoarea Q pentru actiunea executata
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)
        
        # definim functia de pierdere ca si huber_loss
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        # vom folosi optimizatorul Adam
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

atari = gym.make('RiverraidDeterministic-v4')
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, 1024)

init = tf.global_variables_initializer()
saver = tf.train.Saver()    
atari.reset()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "/home/valentinpuiu/projects/deep-q-network/results/river-raid/my_model-17246803") # activeaza daca reiei un model
    filters = sess.run('mainDQN/conv1/kernel:0')
    print(filters.shape)

    filter_cnt=0
    fig, axs = plt.subplots(int(filters.shape[2]*2), int(filters.shape[3]/2), figsize=(8,8))

    for i in range(filters.shape[3]):
        filt=filters[:,:,:, i]
        for j in range(filters.shape[2]):
            axs[int(filter_cnt/(int(filters.shape[3]/2))), int(filter_cnt%(int(filters.shape[3]/2)))].imshow(filt[:,:, j])
            axs[int(filter_cnt/(int(filters.shape[3]/2))), int(filter_cnt%(int(filters.shape[3]/2)))].axis('off')
            filter_cnt+=1
    plt.show()
    rows = sample(range(filters.shape[3]), 8)
    cols = sample(range(filters.shape[2]), 4)
    filter_cnt=0
    for i in range(8):
        plt.imshow(filters[:,:, cols[i % 4], rows[i]])
        name = 'conv-1-' + str(rows[i]) + '-' + str(cols[i % 4]) + '.png'
        plt.savefig(name)
        filter_cnt+=1

### al doilea strat

    filters = sess.run('mainDQN/conv2/kernel:0')
    print(filters.shape)
    rows = sample(range(filters.shape[3]), 8)
    cols = sample(range(filters.shape[2]), 8)
    filter_cnt=0
    for i in range(8):
        plt.imshow(filters[:,:, cols[i], rows[i]])
        name = 'conv-2-' + str(rows[i]) + '-' + str(cols[i]) + '.png'
        plt.savefig(name)
        filter_cnt+=1

## al treilea strat

    filters = sess.run('mainDQN/conv3/kernel:0')
    print(filters.shape)
    rows = sample(range(filters.shape[3]), 8)
    cols = sample(range(filters.shape[2]), 8)
    filter_cnt=0
    for i in range(8):
        plt.imshow(filters[:,:, cols[i], rows[i]])
        name = 'conv-3-' + str(rows[i]) + '-' + str(cols[i]) + '.png'
        plt.savefig(name)
        filter_cnt+=1

    new_state = np.zeros((84,84,4))
    frame_processor = FrameProcessor()
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)
    new_frame, reward, done, info = atari.step(1)
    processed_new_frame = frame_processor(sess, new_frame)
    new_state = np.append(new_state[:, :, 1:], processed_new_frame, axis=2)

    activations = sess.run(MAIN_DQN.conv1, feed_dict={MAIN_DQN.input:[new_state]})

    fig, axs = plt.subplots(int(activations.shape[0]*4), int(activations.shape[3]/4), figsize=(20,20))
    activ_cnt=0
    for i in range(activations.shape[3]):

        activ=activations[:,:,:, i]

        for j in range(activations.shape[0]):
            axs[int(activ_cnt/(int(activations.shape[3]/4))), int(activ_cnt%(int(activations.shape[3]/4)))].imshow(activ[j,:, :])
            axs[int(activ_cnt/(int(activations.shape[3]/4))), int(activ_cnt%(int(activations.shape[3]/4)))].axis('off')
            activ_cnt+=1
    plt.show()
    activ_cnt = 0
    print(activations.shape)
    rows = sample(range(activations.shape[3]), 8)

    for i in range(8):
        plt.imshow(activations[0, :,:, rows[i]])
        name = 'activ-1-' + str(rows[i]) + '-' + str(0) + '.png'
        plt.savefig(name)
    
#### al doilea strat

    activations = sess.run(MAIN_DQN.conv2, feed_dict={MAIN_DQN.input:[new_state]})
    print(activations.shape)
    rows = sample(range(activations.shape[3]), 8)
    for i in range(8):
        plt.imshow(activations[0, :,:, rows[i]])
        name = 'activ-2-' + str(rows[i]) + '-' + str(0) + '.png'
        plt.savefig(name)
    
#### al treilea strat

    activations = sess.run(MAIN_DQN.conv3, feed_dict={MAIN_DQN.input:[new_state]})
    print(activations.shape)
    rows = sample(range(activations.shape[3]), 8)
    for i in range(8):
        plt.imshow(activations[0, :, :, rows[i]])
        name = 'activ-3-' + str(rows[i]) + '-' + str(0) + '.png'
        plt.savefig(name)
