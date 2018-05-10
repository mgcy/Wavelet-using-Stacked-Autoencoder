# perform wavelet using autoencoder

import numpy as np
import tensorflow as tf
import pywt
import scipy.misc
import matplotlib.pyplot as plt
import time
import datetime as dt
from wavelet_autoencoder_helper import *

# monitor start time
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))

# extract data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_train_samples = 6000
num_test_samples = 10

#########################################################################
# wavelet transformation
# wavelet parameters
wavelet_level = 1
new_dim = int(28/(2 ** (wavelet_level)))

# wavelet transform for original images
original_image = []
wavelet_train_comp_image =  np.zeros([num_train_samples,new_dim ** 2],
                                dtype='float32')
wavelet_train_rec_image = np.zeros([num_train_samples, 784],
                                dtype='float32')
wavelet_test_comp_image =  np.zeros([num_train_samples,new_dim ** 2],
                                dtype='float32')
wavelet_test_rec_image = np.zeros([num_train_samples, 784],
                                dtype='float32')

# 1-level wavelet decompression and reconstruction
# preprocess training data
for i in range(0, num_train_samples):
    original_image = np.reshape(mnist.train.images[i],(28,28))
    cA, (cH, cV, cD) = pywt.wavedec2(
        data = original_image,
        wavelet = 'haar',
        level = wavelet_level)
    
    wavelet_train_comp_image[i] = np.reshape(cA, (new_dim ** 2))

    train_rec_image = pywt.waverec2(coeffs = [cA, (cH, cV, cD)],
                              wavelet = 'haar')
    wavelet_train_rec_image[i] = np.reshape(train_rec_image, (784))
    
# preprocess testing data
for i in range(0, num_test_samples):
    original_image = np.reshape(mnist.test.images[i],(28,28))
    cA, (cH, cV, cD) = pywt.wavedec2(
        data = original_image,
        wavelet = 'haar',
        level = wavelet_level)
    
    wavelet_test_comp_image[i] = np.reshape(cA, (new_dim ** 2))

    test_rec_image = pywt.waverec2(coeffs = [cA, (cH, cV, cD)],
                              wavelet = 'haar')
    wavelet_test_rec_image[i] = np.reshape(test_rec_image, (784))

#display some wavelet decomposition
n = 9
canvas1_orig = np.empty((28 * n, 28 * n))
canvas1_comp = np.empty((14 * n, 14 * n))
for i in range(n):
    # Display original images
    for j in range(n):
        # Draw the original digits
        canvas1_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            mnist.test.images[j].reshape([28, 28])
    # Display reconstructed images
    for j in range(n):
        # Draw the reconstructed digits
        canvas1_comp[i * 14:(i + 1) * 14, j * 14:(j + 1) * 14] = \
            wavelet_test_comp_image[j].reshape([14, 14])
'''        
print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas1_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas1_comp, origin="upper", cmap="gray")
plt.show()
'''
# show a digit
#scipy.misc.imsave('wavelet_test_comp_image.png', cA)
#scipy.misc.imsave('wavelet_test_rec_image.png', test_rec_image)    


for i in range(9):
    # original images
    plt.subplot(5, 9, i+1)
    plt.imshow(mnist.test.images[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
 
######################################################################
# built autoencoder
# Training Parameters
learning_rate = 0.1
num_steps = 5000

display_step = 1

# Network Parameters
num_hidden_1 = 441 # 1st layer num features
num_hidden_2 = 196 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X1 = tf.placeholder("float", [None, num_input])
X2 = tf.placeholder("float", [None, 196])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}
# Building the encoder
def encoder(x):
    # Encoder Hidden layer 1 with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer 2 with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X1)
decoder_op = decoder(X2)

# Prediction
y_pred = encoder_op
# Targets (Labels) are the input data.
y_true = X2

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
loss1=[]

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={
            X1: mnist.train.images[:num_train_samples],
            X2: wavelet_train_comp_image})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            loss1.append(l)
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 9
    
    g1 = sess.run(encoder_op, feed_dict={
        X1: mnist.test.images[:9],
        X2: wavelet_test_comp_image[:9]})
    
    '''
    canvas2_orig = np.empty((28 * n, 28 * n))
    canvas2_recon = np.empty((14 * n, 14 * n))
    for i in range(n):
        # MNIST test set
        # Encode and decode the digit image
        #g = sess.run(encoder_op, feed_dict={
            #X1: mnist.train.images[:num_train_samples],
            #X2: wavelet_train_comp_image})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas2_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                mnist.train.images[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas2_recon[i * 14:(i + 1) * 14, j * 14:(j + 1) * 14] = \
                g1[j].reshape([14, 14])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas2_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas2_recon, origin="upper", cmap="gray")
    plt.show()
    '''

#########################################################################
# built autodecoder


# Network Parameters
num_hidden_1 = 441 # 1st layer num features
num_hidden_2 = 784 # 2nd layer num features (the latent dim)
num_input = 196 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X1 = tf.placeholder("float", [None, num_input])
X2 = tf.placeholder("float", [None, 784])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}
# Building the encoder
def encoder(x):
    # Encoder Hidden layer 1 with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer 2 with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X1)
decoder_op = decoder(X2)

# Prediction
y_pred = encoder_op
# Targets (Labels) are the input data.
y_true = X2

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
loss2=[]

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        #batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={
            X1: wavelet_train_comp_image,
            X2: wavelet_train_rec_image})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            loss2.append(l)
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    g2 = sess.run(encoder_op, feed_dict={
        X1: wavelet_test_comp_image[:9],
        X2: wavelet_test_rec_image[:9]})
    
    n = 9
    canvas3_orig = np.empty((14 * n, 14 * n))
    canvas3_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        # Encode and decode the digit image
        #g = sess.run(encoder_op, feed_dict={
            #X1: mnist.train.images[:num_train_samples],
            #X2: wavelet_train_comp_image})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas3_orig[i * 14:(i + 1) * 14, j * 14:(j + 1) * 14] = \
                wavelet_test_comp_image[j].reshape([14, 14])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas3_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g2[j].reshape([28, 28])
    '''
    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas3_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas3_recon, origin="upper", cmap="gray")
    plt.show()
    '''

######################################################################
# show some outputs
for i in range(9):
    '''
    # original images
    plt.subplot(5, 9, i+1)
    plt.imshow(mnist.train.images[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
    '''
    
    # wavelet decompostion
    plt.subplot(5, 9, i+1+9)
    plt.imshow(wavelet_test_comp_image[i].reshape(14, 14))
    plt.gray()
    plt.axis('off')

    # autoencoder decompostion
    plt.subplot(5, 9, i+1+9+9)
    plt.imshow(g1[i].reshape(14, 14))
    plt.gray()
    plt.axis('off')
       
    # wavelet reconstruction
    plt.subplot(5, 9, i+1+9+9+9)
    plt.imshow(wavelet_test_rec_image[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')

    # autoencoder reconstruction
    plt.subplot(5, 9, i+1+9+9+9+9)
    plt.imshow(g2[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')

plt.show()

######################################################################
# show loss plot
plt.plot(range(len(loss1)),loss1,'r',label='Autoencoder Loss')
plt.plot(range(len(loss2)),loss2,'b',label='Autodecoder Loss')
plt.title('Training Loss')
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.legend()
plt.show()

######################################################################
# calculate snr
# decomposition snr
# num_test_samples = 10

comp_sum_snr = 0
for i in range(num_test_samples):
    comp_sum_snr = comp_sum_snr + snr(wavelet_test_comp_image[i], g1[i])
comp_snr = comp_sum_snr/num_test_samples
# reconstuction snr
rec_sum_snr = 0
for i in range(num_test_samples):
    rec_sum_snr = rec_sum_snr + snr(wavelet_test_rec_image[i], g2[i])
rec_snr = rec_sum_snr/num_test_samples
print('Decomposition snr = '+ str(comp_snr))
print('Reconstruction snr = '+ str(rec_snr))

# calculate mse
# decomposition mse
comp_sum_mse = 0
for i in range(num_test_samples):
    comp_sum_mse = comp_sum_mse + mse(wavelet_test_comp_image[i], g1[i])
comp_mse = comp_sum_mse/num_test_samples
# reconstuction mse
rec_sum_mse = 0
for i in range(num_test_samples):
    rec_sum_mse = rec_sum_mse + mse(wavelet_test_rec_image[i], g2[i])
rec_mse = rec_sum_mse/num_test_samples
print('Decomposition mse = '+ str(comp_mse))
print('Reconstruction mse = '+ str(rec_mse))


####################################################################
# monitor end time
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

