from mnist import MNIST
import time

from pylearn.neural_network import Network

start_time = time.time()


def preprocess_data(images, labels):
    return ([[x / 255 for x in x_row] for x_row in images],
            [[1 if y == i else 0 for i in range(10)] for y in labels])

mndata = MNIST('.\data\\numbers')
mndata.load_training()
mndata.load_testing()

# mndata.train_labels = mndata.train_labels[0:10000]
# mndata.train_images = mndata.train_images[0:10000]

# mndata.test_labels = mndata.test_labels[0:1000]
# mndata.test_images = mndata.test_images[0:1000]

test_images_wrap, test_labels_wrap = preprocess_data(
    mndata.test_images, mndata.test_labels)
train_images_wrap, train_labels_wrap = preprocess_data(
    mndata.train_images, mndata.train_labels)

print('Training set size:', len(train_images_wrap))

net = Network([784, 60, 10])
net.train_network(train_images_wrap, train_labels_wrap,
                  (test_images_wrap, test_labels_wrap))

print('Done\n')
print("--- %s seconds ---" % (time.time() - start_time))
