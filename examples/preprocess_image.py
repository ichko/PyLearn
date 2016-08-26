from PIL import Image


def image_to_vector(path):
    im = Image.open(path)
    result = [p[0] / 50 for p in list(im.getdata())]
    return result
