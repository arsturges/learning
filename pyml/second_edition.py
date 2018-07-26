# First steps with TensorFlow

import tensorflow as tf

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name='x')
    w = tf.placeholder(dtype=tf.float32, name='w')
    b = tf.Variable(0.7, name='bias')
    z = w * x + b

    init = tf.global_variables_initializer()

with tf.Session(graph=g) as sess:
    sess.run(init)
    print('x={} -- > z={}'.format(x, sess.run(z, feed_dict={x: [1.0, 0.6, -1.8], w:2.0})))

# Working with array structures
import tensorflow as tf
import numpy as np
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name='input_x')
    x2 = tf.reshape(x, shape=(-1, 6), name='x2')
    xsum = tf.reduce_sum(x2, axis=0, name='col_sum')
    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:
    x_array = np.arange(12).reshape(2, 2, 3)
    print('input shape: ', x_array.shape)
    print('Reshaped:\n', sess.run(x2, feed_dict={x: x_array}))
    print('Col sums\n', sess.run(xsum, feed_dict={x: x_array}))
    print('Col means\n', sess.run(xmean, feed_dict={x: x_array}))

# Developing a simple model with the low-level TensorFlow API

import tensorflow as tf
import numpy as np

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

class TfLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        self.X = tf.placeholder(tf.float32, shape=(None, self.x_dim), name='x_input')
        self.y = tf.placeholder(tf.float32, shape=(None), name='y_input')
        print(self.X)
        print(self.y)
        w = tf.Variable(tf.zeros(shape=(1)), name='weight')
        b = tf.Variable(tf.zeros(shape=(1)), name='bias')
        print(w)
        print(b)
        self.z_net = tf.squeeze(w * self.X + b, name='z_net')
        print(self.z_net)
        sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)

def train_linreg(sess, model, X_train, y_train, num_epochs=10):
    sess.run(model.init_op)
    training_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost], feed_dict={model.X: X_train, model.y: y_train})
        training_costs.append(cost)
    return training_costs

lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)
sess = tf.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)

import matplotlib.pyplot as plt
plt.plot(range(1, len(training_costs) + 1), training_costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training cost')
#plt.show()

def predict_linreg(sess, model, X_test):
    return sess.run(model.z_net, feed_dict={model.X: X_test})

plt.scatter(X_train, y_train, marker='s', s=50, label='Training Data')
x_values = range(X_train.shape[0])
y_values = predict_linreg(sess, lrmodel, X_train)
plt.plot(x_values, y_values, color='gray', marker='o', markersize=6, linewidth=3, label='LinReg Model')
#plt.show()

# Building multilayer neural networks using TensorFlow's Layers API

# Obtaining the MNIST data
import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    train_labels_path = os.path.join(path, 'train-labels-idx1-ubyte')
    train_images_path = os.path.join(path, 'train-images-idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    if kind == 'train':
        labels_path = train_labels_path
        images_path = train_images_path
    else:
        labels_path = test_labels_path
        images_path = test_images_path
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
        return images, labels

X_train, y_train = load_mnist("/Users/asturges/learning/kaggle/MNIST_data/", kind='train')
X_test, y_test = load_mnist("/Users/asturges/learning/kaggle/MNIST_data/", kind='t10k')

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val

del X_train, X_test
n_features = X_train_centered.shape[1]
n_classes = 10

random_seed = 123
np.random.seed(random_seed)

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name='tf_y')
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)

    h1 = tf.layers.dense(inputs=tf_x, units=50, activation=tf.tanh, name='layer1')
    h2 = tf.layers.dense(inputs=h1, units=50, activation=tf.tanh, name='layer2')
    logits = tf.layers.dense(inputs=h2, units=10, activation=None, name='layer3')

    predictions = {'classes': tf.argmax(logits, axis=1, name='predicted_classes'), 'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

with g.as_default():
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=cost)
    init_op = tf.global_variables_initializer()

def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)
    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, : - 1]
        y_copy = data[:, -1]
    for i in range(0, X.shape[0], batch_size):
        yield(X_copy[i:i + batch_size, :], y_copy[i:i + batch_size])

"""sess = tf.Session(graph=g)
sess.run(init_op)
for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(X_train_centered, y_train, batch_size=64)
    for batch_X, batch_y in batch_generator:
        feed = {tf_x: batch_X, tf_y: batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(' -- Epoch {}  Avg. Training loss: {}'.format(epoch+1, np.mean(training_costs)))

feed = {tf_x: X_test_centered}
y_pred = sess.run(predictions['classes'], feed_dict=feed)
print('Test Accuracy: {}'.format(100 * np.sum(y_pred == y_test) / y_test.shape[0]))
"""

# Developing a multilayer neural network with Keras
from tensorflow import keras
y_train_onehot = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=50, input_dim=X_train_centered.shape[1], kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))
model.add(keras.layers.Dense(units=50, input_dim=50, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))
model.add(keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=50, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax'))
sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')

history = model.fit(X_train_centered, y_train_onehot, batch_size=64, epochs=50, verbose=1, validation_split=0.1)
y_train_pred = model.predict_classes(X_train_centered, verbose=0)
print('First 3 prediction: {}'.format(y_train_pred[:3]))
correct_predictions = np.sum(y_train == y_train_pred, axis=0)
train_accuracy = correct_predictions / y_train.shape[0]
print('Training accuracy: {}%'.format(train_accuracy * 100))
y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_predictions = np.sum(y_test == y_test_pred)
test_accuracy = correct_predictions / y_test.shape[0]
print('Test accuracy: {}%'.format(test_accuracy * 100))
