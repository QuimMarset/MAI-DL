import cv2
from skimage import img_as_ubyte, img_as_float32
from utils.paths_utils import join_path



def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def image_to_float(image):
    return img_as_float32(image)


def image_to_uint8(image):
    return img_as_ubyte(image)