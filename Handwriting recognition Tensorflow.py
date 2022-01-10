import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mnist = input_data.read_data_sets(".\MNIST", one_hot=True)

''' One Hot data structure
0 = [1,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0] 
...
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
n_pixels = 784
batch_size = 100   # bacth of images  Tensorflow manages batches. There will be 550 batches => 55,000 images for each epoch


# height x width, 28 x 28
x = tf.placeholder('float',[None, n_pixels])    # input: one-dimentional matrix. 784 pixels on each image Each pixel is an entry.
y = tf.placeholder('float')                         

def neural_network_model(x):   # data is the input, the value of each of the 784 pixels. Function returns output (1 to 10)
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_pixels, n_nodes_hl1])),    # weights between each input pixel and each layer 1 cell
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}  
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), # weights between layer 1 cells and layer 2 cells
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}   
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), # weights between layer 2 cells and layer 3 cells
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}   
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),     # weights between layer 3 cells and outputs (10 outputs)
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # input_data * weights + biases
    l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])      # layer 1 = relu (input data * weights1 + biases1)
    l1 = tf.nn.relu(l1)   
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])        # layer 2 = relu (layer 1  * weights2 + biases2)
    l2 = tf.nn.relu(l2)    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])        # layer 3 = relu (layer 2 * weights3 + biases3)
    l3 = tf.nn.relu(l3)
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']               # output = layer3 * ouput weights + output biases

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)      # learning_rate = 0.001

    n_epochs = 10   # cyles of feed forward and back propagation (training sessions)
    n_dataelements = int(mnist.train.num_examples/batch_size)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(n_dataelements):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, 'completed out of ', n_epochs, ' Batches: ', n_dataelements, ' loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)




                



        








    
    
