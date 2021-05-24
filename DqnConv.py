import tensorflow._api.v2.compat.v1 as tf

class DQN(object):
    """Implementeaza o retea deep-Q duelata"""
    
    # pylint: disable=too-many-instance-attributes
    
    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001, screen_height=84, screen_width=84, number_of_frames=4):
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