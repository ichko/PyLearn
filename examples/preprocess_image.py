import sys

import matplotlib.pyplot as plt
from PIL import Image


def image_to_vector(path):
    im = Image.open(path)
    result = [(255 - p[0]) / 255 for p in list(im.getdata())]
    return result


def show_image(image):
    image_matrix = [[[int(p * 255)] * 3
                    for p in image[k * 28:(k + 1) * 28]]
                    for k in range(28)]
    plt.imshow(image_matrix)
    plt.show()


def print_image(image):
    for i in range(28 * 28):
        sys.stdout.write(' ' if image[i] < 0.5 else '#')
        if(i % 28 == 0):
            sys.stdout.write('\n')
