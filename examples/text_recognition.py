from mnist import MNIST
import time
import sys

from pylearn.neural_network import NeuralNetwork
from pylearn.preprocess import InputData


mndata = MNIST('.\data\\numbers')
mndata.load_training()
mndata.load_testing()

mndata.train_labels = mndata.train_labels[0:1000]
mndata.train_images = mndata.train_images[0:1000]

mndata.test_labels = mndata.test_labels[0:50]
mndata.test_images = mndata.test_images[0:50]

test_images_wrap, test_labels_wrap = InputData.image_matrix_normalizer(
    mndata.test_images, mndata.test_labels, range(10))
train_images_wrap, train_labels_wrap = InputData.image_matrix_normalizer(
    mndata.train_images, mndata.train_labels, range(10))

print('Training set size:', len(train_images_wrap))


def notifier(net, epoch_id):
    print("Epoch {0}: {1} / {2}, cost: {3}".format(
          epoch_id + 1,
          net.test_network(test_images_wrap, test_labels_wrap),
          len(test_images_wrap),
          net.batch_cost(test_images_wrap, test_labels_wrap)))


start_time = time.time()

net = NeuralNetwork([784, 20, 10], 3)
net.batch_size = 10
net.max_iterations = 15
net.epoch_end_notifier = notifier

net.fit(train_images_wrap, train_labels_wrap)

print("--- %s seconds ---" % (time.time() - start_time))
