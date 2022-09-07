import tensorflow as tf
from tensorflow import keras
import os
from utils.file_io_utils import write_json_string



class CNNModel(tf.keras.Model):
    
    def __init__(self, image_size=256):
        super().__init__()
        
        self.image_size = image_size
        self.conv_layer = keras.layers.Conv2D(1, (1, 1), padding="same", activation="linear")



    def call(self, data):
        frames = data[..., :3]
        masks = tf.expand_dims(data[..., 3], axis=-1)

        c1, p1 = self.downscale_blocks[0](frames)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)

        return self.conv_layer(u4) * masks

    
    def save_architecture(self, save_path):
        model_json_string = self.to_json()
        write_json_string(model_json_string, os.path.join(save_path, 'model_architecture.json'))


    def load_model_weights(self, load_path):
        self.load_weights(os.path.join(load_path, 'model_weights'))


    def save_model_weights(self, save_path):
        self.save_weights(os.path.join(save_path, 'model_weights'))