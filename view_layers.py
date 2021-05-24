import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import helpers
import numpy as np
from ddqn_orig import DQN
env = gym.make('BreakoutDeterministic-v4')
a = helpers.preprocess(env.reset())
print(a.shape)
plt.imshow(a, cmap='gray')
plt.show()

params = {}
params.update({'space_width': 210})
params.update({'space_height': 160})
params.update({'space_color': 3})
params.update({'space_frames': 4})
params.update({'state_memory_length': 10000})
params.update({'name': 'classic-model'})
params.update({'number_of_steps': 5000000})
params.update({'learning_step': 1000})
params.update({'saving_episode': 5})
params.update({'episodes': 20000})
params.update({'gamma': 0.99})
params.update({'updating_network_step': 10})
params.update({'start_to_learn': 1000})
params.update({'batch_size': 4})
params.update({'min_epsilon': 0.07})
params.update({'epsilon': 1})
params.update({'epsilon_decay_steps': 1000000})
params.update({'learning_rate': 0.0001})
params.update({'momentum': 0.01})
params.update({'rho': 0.01})
params.update({'beta1': 0.01})
params.update({'beta2': 0.01})
params.update({'learning_rate_power': -0.01})
params.update({'initial_accumulator_value': 0.01})
params.update({'show_preview': True})
params.update({'when_to_show': 50})
params.update({'minimum_reward' :-200})
params.update({'optimizer': 'RMSprop'})
params.update({'loss': 'mse'})

test_model = DQN(env.action_space.n, 1024, 0.000001)
layer_names = [layer.name for layer in test_model.layers]
layer_outputs = [layer.output for layer in test_model.layers]
print(layer_names)
print(layer_outputs)

feature_map_model = tf.keras.models.Model(inputs=test_model.input, outputs=layer_outputs)

def plotFilters(conv_filter):
    if (conv_filter.shape[2] > 3):
        fig, axes = plt.subplots(1, 3, figsize=(5,5))

        axes = axes.flatten()
        for img, ax in zip( conv_filter, axes):
            ax.imshow(img)
            ax.axis('off')
    else:
        plt.imshow(conv_filter)
    plt.tight_layout()
    plt.show()

for layer in test_model.q_model.layers:
    if 'conv' in layer.name:
        filters, bias= layer.get_weights()
        print(layer.name, filters.shape)
         #normalize filter values between  0 and 1 for visualization
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)  
        print(filters.shape[3])
        axis_x=1
        #plotting all the filters
        for i in range(filters.shape[3]):
        #for i in range(6):
            #get the filters
            filt=filters[:,:,:, i]
            plotFilters(filt)

#Visualizing the filters
#plt.figure(figsize=(5,5))

for layer in test_model.q_model.layers:
    if 'conv' in layer.name:
        weights, bias= layer.get_weights()
        print(layer.name, weights.shape)
         #normalize filter values between  0 and 1 for visualization
        f_min, f_max = weights.min(), weights.max()
        filters = (weights - f_min) / (f_max - f_min)  
        print(weights.shape[3])
        filter_cnt=1
        #plotting all the filters
        print('Le filters shape: ',filters.shape)
        for i in range(filters.shape[3]):
        #for i in range(6):
            #get the filters
            filt=filters[:,:,:, i]
            #plotting ecah channel
            for j in range(filters.shape[2]):
                #plt.figure( figsize=(5, 5) )
                #f = plt.figure(figsize=(10,10))
                ax= plt.subplot(filters.shape[3], filters.shape[2], filter_cnt  )
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(filt[:,:, j])
                filter_cnt+=1
        plt.show()

# img_path='C:\\Data\\CV\\dogs-vs-cats\\test1\\136.jpg' #dog
# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in test_model.q_model.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = test_model.q_model.input, outputs = successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
#cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
#dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

#img_path = random.choice(cat_img_files + dog_img_files)

# img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

# x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x = a.astype('float')
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in test_model.q_model.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(layer_name, ' : ', feature_map.shape)
    if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
        if (len(feature_map.shape) == 4):
            x  = feature_map[0, :, :, i]
        print(x.shape)

        x = x.astype('float')
        x -= x.mean()

        if (x.std() == 0):
            x = x
        else:
            x /= x.std()
        x *=  64
        x += 128
        x  = np.clip(x, 0, 255).astype('uint8')
        display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
    plt.show()