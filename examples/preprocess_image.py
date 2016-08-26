from PIL import Image


def image_to_vector(path):
    im = Image.open(path)
    result = [(255 - p[0]) / 255 for p in list(im.getdata())]
    return result
