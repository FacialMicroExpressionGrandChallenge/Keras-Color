# Art Style Transfer
# Setting
import scipy.io
import scipy.misc
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from pprint import pprint

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import os
import sys

from PIL import Image

import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions



# 모델 로드, 저장 방법
# model = VGG16(weights='imagenet', include_top=False)
# model.save('C:/Users/YY/Documents/Winter Project/Spring/art_vgg16.h5')
# vgg = load_model('C:/Users/YY/Documents/Winter Project/Spring/art_vgg16.h5')
# layers = dict([(layer.name, layer.output) for layer in model.layers])
# print(layers)
# model.count_params()


# Load the Content/Style Image
height, width = 256, 256
content_image_path = 'C:/Users/YY/Documents/Data/NN/StyleTransfer/Target/k4.jpg'
content_image = Image.open(content_image_path)
content_image = content_image.resize((width, height))

"""
imshow(content_image)
plt.show()
"""

style_image_path = 'C:/Users/YY/Documents/Data/NN/StyleTransfer/Style/picasso.jpg'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))

content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)   # (512,512,3) -> (1,512,512,3)
print(content_array.shape)

style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)
print(style_array.shape)

# Preprocess Image
def preprocess_by_subtract_mean_of_rgb(image_array):
    image_array[:, :, :, 0] -= 103.939
    image_array[:, :, :, 1] -= 116.779
    image_array[:, :, :, 2] -= 123.68
    return image_array[:, :, :, ::-1]

content_array = preprocess_by_subtract_mean_of_rgb(content_array)
style_array = preprocess_by_subtract_mean_of_rgb(style_array)

# 이미지 데이터를 케라스 변수로 변환
content_image = K.variable(content_array, name="Content_BackendVariable")
style_image = K.variable(style_array, name="Style_BackendVariable")
combination_image = K.placeholder(shape=(1, height, width, 3), name="Combination_Placeholder")
input_tensor = K.concatenate([content_image, style_image, combination_image], axis=0)

# input_tensor shape: 3, 512, 512, 3


# Reuse a model pre-trained for image classification to define loss functions
# Option: include_top: no Fully_connected layers
model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
model.compile(optimizer='adam', loss='categorical_crossentropy')

layers = dict([(layer.name, layer.output) for layer in model.layers])
pprint(layers)
# model.save('C:/Users/YY/Documents/Data/NN/Model/art_vgg16.h5')
# model.summary()




# Hyper-parameters
content_weight = 0.05
style_weight = 5.0
total_variation_weight = 1.0

# 전체 손실을 담을 변수 선언
loss = K.variable(0.0)

def content_loss(content, combination):
    return K.sum(K.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss = loss + content_weight * content_loss(content_image_features, combination_features)





def gram_matrix(_x):
    features = K.batch_flatten(K.permute_dimensions(_x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    st = K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
    return st

# 스타일 loss를 계산할 layer
feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3', 'block5_conv3']

# 스타일 loss 계산
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss = loss + (style_weight / len(feature_layers)) * sl

cut = 1

def total_variation_loss(x):
    a = K.square(x[:, :height-cut, :width-cut, :] - x[:, cut:, :width-cut, :])
    b = K.square(x[:, :height-cut, :width-cut, :] - x[:, :height-cut, cut:, :])
    return K.sum(K.pow(a + b, 1.25))

loss = loss + total_variation_weight * total_variation_loss(combination_image)




# Define gradients of the total loss relative to the combination image
# -> Use these gradients to iteratively improve upon our combination image to minimise the loss.
grads = K.gradients(loss, combination_image)




outputs = [loss]
outputs = outputs + grads
f_outputs = K.function([combination_image], outputs)



def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values



class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


# 노이즈 이미지 생성
x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.0

# 학습
iterations = 20


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(func=evaluator.loss,
                                     x0=x.flatten(),
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))




x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

result = Image.fromarray(x)
result.save('C:/Users/YY/Documents/result.bmp')

