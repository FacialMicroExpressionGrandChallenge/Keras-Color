"""
Color Detection with Tiny-Darknet
"""

from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import os

from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from skimage import data, io, graph, segmentation, color
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import colorsys

from sklearn.cluster import KMeans


#1 데이터 로드
path = 'C:/Users/YY/Documents/Data/CCP/Color'

train_datagen = ImageDataGenerator(rescale= 1./255)
test_datagen = ImageDataGenerator(rescale= 1./255)

train = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=32, class_mode='categorical')
test = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=32, class_mode='categorical')


def LoadPreprocess(path, nb_classes):
    dataset = []
    y = []

    # classes_list: path내에 있는 폴더명을 리스트로 만듦
    classes_list = os.listdir(path)

    for class_number, class_name in enumerate(classes_list):
        # img_list: 'rock.jpg'과 같은 이미지명을 담은 리스트임
        img_list = os.listdir(os.path.join(path, class_name))

        for img_number, img_name in enumerate(img_list):
            # 각각의 이미지의 path name을 정의한다.
            img_path_name = os.path.join(path, class_name, img_name)

            # 이미지 로드
            img = image.load_img(img_path_name, target_size=(224, 224))

            # Superpixel화
            img = image.img_to_array(img) / 255.

            # n_segments: 몇 개의 구역으로 나누고 싶은가?
            labels = segmentation.slic(img, compactness=30, n_segments=30)
            labels = labels + 1
            regions = regionprops(labels)
            img_input = color.label2rgb(labels, img, kind='avg')

            # 이미지를 np.array로 바꿔줌
            # shape은 (height, width, channels) = (128, 128, 3)
            dataset.append(img_input)
            y.append(class_number)

    dataset = np.array(dataset)
    y = np.array(y)
    Y = np.eye(nb_classes)[y.astype(int)]
    dataset = dataset.astype('float32')

    return dataset, Y




#2 모델링
class ColorCNN(Model):
    def __init__(self, in_shape, nb_classes):
        self.in_shape = in_shape
        self.nb_classes = nb_classes
        self.BuildModel()
        super().__init__(self.X_input, self.Y)
        self.compile()

    def BuildModel(self):
        in_shape = self.in_shape
        nb_classes = self.nb_classes
        X_input = Input(shape=in_shape)
        
        # Stage1
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same')(X_input)
        X = MaxPool2D(pool_size=(2, 2), strides=2)(X)
        X = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(X)
        X = MaxPool2D(pool_size=(2, 2), strides=2)(X)
        
        # Stage2        
        X = Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding='valid')(X)
        X = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(X)
        X = Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding='valid')(X)
        X = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(X)
        X = MaxPool2D(pool_size=(2, 2), strides=2)(X)

        # Stage3
        X = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='valid')(X)
        X = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(X)
        X = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='valid')(X)
        X = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(X)
        X = MaxPool2D(pool_size=(2, 2), strides=2)(X)

        # Stage4
        X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='valid')(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(X)
        X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='valid')(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(X)
        X = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='valid')(X)
        X = Conv2D(filters=1000, kernel_size=(1, 1), strides=1, padding='valid')(X)

        # Stage5
        X = AveragePooling2D(pool_size=(14, 14))(X)
        X = Flatten()(X)
        Y = Dense(units=nb_classes, activation='softmax')(X)

        self.X_input, self.Y = X_input, Y

    def compile(self):
        Model.compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#3 학습
X, Y = LoadPreprocess(path=path, nb_classes=3)
model = ColorCNN(in_shape=(224, 224, 3), nb_classes=3)

model.fit_generator(train, steps_per_epoch=300, epochs=1, validation_data=test, validation_steps=2000)
# model.summary()

epochs=1
history = model.fit(X, Y, batch_size=32, epochs=epochs, shuffle=True,
                    validation_split=0.1, callbacks=[EarlyStopping(patience=10)])
model.save('Color_epoch25.h5')



#4 테스트
test_path = 'C:/Users/YY/Documents/Data/CCP/Test'
X_test, Y_test = LoadPreprocess(path=test_path, nb_classes=3)
prediction = model.predict(X_test)

P = np.argmax(prediction, axis=1)
T = np.argmax(Y_test, axis=1)

print("Test Accuracy: ", np.sum(P==T)/X_test.shape[0])



#----------------------
#image_path = 'C:/Users/YY/Documents/target.jpg'
#img = img_as_float(io.imread(image_path))
#plt.imshow(image)
#segments = slic(img, n_segments=3, sigma=5)
#combined = mark_boundaries(img, segments, (0, 0, 0))    # same shape as image shape
#plt.imshow(combined)
#plt.show()

image_path = 'C:/Users/YY/Documents/Data/CCP/Test/Blue/b1.jpg'
img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)/255.


# n_segments: 몇 개의 구역으로 나누고 싶은가?
labels = segmentation.slic(img, compactness=30, n_segments=5000)
labels = labels + 1
regions = regionprops(labels)
input_img = color.label2rgb(labels, img, kind='avg')

plt.imshow(input_img)
plt.show()

f = input_img

R = np.unique(f[:,:,0])
G = np.unique(f[:,:,1])
B = np.unique(f[:,:,2])
UniqueRGB = np.stack([R, G, B], axis=1)
print(UniqueRGB.shape)    # 행은 색깔 수, 열은 R,G,B


def RGBtoHSV(array):
    UniqueHSV = np.zeros(array.shape)

    for i in range(array.shape[0]):
        r, g, b = array[i]
        hsv = colorsys.rgb_to_hsv(r, g, b)
        unit = np.array(hsv)
        UniqueHSV[i] = unit

    return UniqueHSV


UniqueHSV = RGBtoHSV(UniqueRGB)
sv = colorsys.rgb_to_hsv(UniqueRGB)
X = UniqueHSV

kmeans = KMeans(n_clusters=3, n_jobs=-1)
kmeans.fit(X)
centers = kmeans.cluster_centers_




