from mnist import MNIST
import time

from pylearn.neural_network import Network

start_time = time.time()


def preprocess_data(images, labels):
    return ([[x for x in x_row] for x_row in images],
            [[1 if y == i else 0 for i in range(10)] for y in labels])

mndata = MNIST('.\data\\numbers')
mndata.load_training()
mndata.load_testing()

test_images_wrap, test_labels_wrap = preprocess_data(
    mndata.test_images, mndata.test_labels)
train_images_wrap, train_labels_wrap = preprocess_data(
    mndata.train_images, mndata.train_labels)

'''
train_labels_wrap = train_labels_wrap[0:30000]
train_images_wrap = train_images_wrap[0:30000]

test_labels_wrap = test_labels_wrap[0:500]
test_images_wrap = test_images_wrap[0:500]
'''

print('Training set size:', len(train_images_wrap))

net = Network([784, 30, 10])
net.train_network(train_images_wrap, train_labels_wrap,
                  (test_images_wrap, test_labels_wrap))

print('Done\n')
print("--- %s seconds ---" % (time.time() - start_time))
