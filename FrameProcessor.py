import tensorflow._api.v2.compat.v1 as tf

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