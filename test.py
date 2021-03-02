#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from tensorflow.keras.layers import Input

from nets.yolo4.yolo4 import yolo_body

if __name__ == "__main__":
    inputs = Input([416,416,3])
    model = yolo_body(inputs,3,80)
    model.summary()

# var = 1
# a = 3
# def fun():
#     global var
#     # var = 2
#     var = var + 1
#     return var
# print(fun())

# import os
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras.backend as K
# from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
#                                         TensorBoard)
# from tensorflow.keras.layers import Input, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# from nets.loss import yolo_loss
# from nets.yolo4_tiny.yolo4_tiny import yolo_body as yolo_body_tiny
# from nets.yolo4.yolo4 import yolo_body
# from utils.utils import (ModelCheckpoint, WarmUpCosineDecayScheduler,
#                          get_random_data, get_random_data_with_Mosaic, rand)

# strategy = tf.distribute.MirroredStrategy()
# image_input = Input(shape=(None, None, 3))
# num_anchors = 4
# num_classes = 80

# with strategy.scope():
#     model = yolo_body_tiny(image_input, num_anchors//2, num_classes)

    # image = tf.keras.layers.Conv2D(32, 3, activation='relu')(image_input)
    # image = tf.keras.layers.MaxPooling2D()(image)
    # image = tf.keras.layers.Flatten()(image)
    # image = tf.keras.layers.Dense(64, activation='relu')(image)
    # image = tf.keras.layers.Dense(10)(image)
    # model = tf.keras.models.Model(image_input,image)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, 3, activation='relu',
    #                            input_shape=(28, 28, 1)),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])

    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(),
    #               metrics=['accuracy'])

    # model.summary()
