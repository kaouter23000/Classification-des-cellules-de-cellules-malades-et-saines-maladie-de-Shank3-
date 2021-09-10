# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:50:47 2021

@author: kberrahal
"""
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from keras.applications.vgg19 import VGG19
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D,MaxPool2D
from keras.optimizers import Adam
import seaborn as sns

import keras
from keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'train'
valid_path = 'validation'

# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False    
  # useful for getting number of classes
folders = glob('train/*')
# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='sigmoid')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)




"""

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="sigmoid"))

"""

model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   #shear_range = 0.2,
                                   #zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                 
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('validation',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')


# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)



acc = r.history['accuracy']
val_acc = r.history['val_accuracy']
loss = r.history['loss']
val_loss = r.history['val_loss']

epochs_range = range(10)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#
labels = ['malade', 'saine']
img_size = 224
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
test=get_data('testt')
m = []
for i in test:
    if(i[1] == 0):
        m.append("malade")
    else:
        m.append("saine")
print (m)
s=[]
for i in m:
    if i== 'malade':
        s.append(0)
    else:
        s.append(1)
print(s)

x_test=[]
y_test=[]

for feature, label in test:
  x_test.append(feature)
  y_test.append(label)







##########################################################

#Classification cellules saines et malades

import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
paths="saine/*.tif"
n=[]
p=[]
name=[]
for file in glob.glob(paths):
        imagename=os.path.basename(file)
        name.append(imagename)
        img=cv2.imread(file) 
        img=cv2.resize(img,(224,224))
        test_image = image.img_to_array(img)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        resultwt=result[0][1]
        p.append(resultwt)
        resulthete=result[0][0]
        n.append(resulthete)
    

print(len(n))
print(len(p))
a = list(range(len(name)))
print(len(n))

import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
paths="malade/*.tif"
r=[]
j=[]
name=[]
for file in glob.glob(paths):
        imagename=os.path.basename(file)
        name.append(imagename)
        img=cv2.imread(file) 
        img=cv2.resize(img,(224,224))
        test_image = image.img_to_array(img)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        resultwt=result[0][1]
        r.append(resultwt)
        resulthete=result[0][0]
        j.append(resulthete)
    
print(len(r))
print(len(j))
t = list(range(len(name)))
print(len(t))
width=0.9
plt.figure(figsize=(15, 15))  
plt.subplot(2, 2, 1)
p1 = plt.bar(a, p, width, color='g')
p2 = plt.bar(a, n, width, bottom=p, color='b')
plt.ylim([0,1])
plt.grid(False)
plt.ylabel('pourcetage de classification')
plt.title('Saine')
plt.subplot(2, 2, 2)
p1 = plt.bar(t, r, width, color='g')
p2 = plt.bar(t, j, width, bottom=r, color='b')
plt.ylim([0,1])
plt.grid(False)
plt.title('Malade')
plt.show()        

##Classification cellules malades testée par le lithium ( molecule pharmacologique)

#**********************************2mm
import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
paths="2/*.tif"
k=[]
m=[]
name=[]
for file in glob.glob(paths):
        imagename=os.path.basename(file)
        name.append(imagename)
        img=cv2.imread(file) 
        img=cv2.resize(img,(224,224))
        test_image = image.img_to_array(img)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        resultwt=result[0][1]
        k.append(resultwt)
        resulthete=result[0][0]
        m.append(resulthete)
    
print(k)
print(m)
print(len(k))
print(len(m))
a = list(range(len(name)))
print(len(a))

#**********************************5mm



import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
paths="5/*.tif"
r=[]
j=[]
name=[]
for file in glob.glob(paths):
        imagename=os.path.basename(file)
        name.append(imagename)
        img=cv2.imread(file) 
        img=cv2.resize(img,(224,224))
        test_image = image.img_to_array(img)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        resultwt=result[0][1]
        r.append(resultwt)
        resulthete=result[0][0]
        j.append(resulthete)
    

print(len(r))
print(len(j))
R = list(range(len(name)))
print(len(R))

#**********************************h2o

import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
paths="h/*.tif"
s=[]
h=[]
name=[]
for file in glob.glob(paths):
        imagename=os.path.basename(file)
        name.append(imagename)
        img=cv2.imread(file) 
        img=cv2.resize(img,(224,224))
        test_image = image.img_to_array(img)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        resultwt=result[0][1]
        s.append(resultwt)
        resulthete=result[0][0]
        h.append(resulthete)
    

print(len(s))
print(len(h))
S= list(range(len(name)))
print(len(S))

#**********************************N2B27
import glob
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
paths="n/*.tif"
d=[]
f=[]
name=[]
for file in glob.glob(paths):
        imagename=os.path.basename(file)
        name.append(imagename)
        img=cv2.imread(file) 
        img=cv2.resize(img,(224,224))
        test_image = image.img_to_array(img)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        resultwt=result[0][1]
        d.append(resultwt)
        resulthete=result[0][0]
        f.append(resulthete)
    

print(len(d))
print(len(f))
D = list(range(len(name)))
print(len(D))


#2
width=0.7
plt.figure(figsize=(15, 15))  
plt.subplot(4, 4, 1)
p1 = plt.bar(a, k, width, color='g')
p2 = plt.bar(a, m, width, bottom=k, color='b')
plt.ylim([0,1])

plt.grid(False)

plt.ylabel('pourcetage de classification')
plt.title('lithium 2mm')
#5
plt.subplot(4, 4, 2)
p1 = plt.bar(R, r, width, color='g')
p2 = plt.bar(R, j, width, bottom=r, color='b')
plt.ylim([0,1])
plt.grid(False)

plt.title('lithium 5mm')

#h2o
plt.subplot(4, 4,3)
p1 = plt.bar(S, s, width, color='g')
p2 = plt.bar(S, h, width, bottom=s, color='b')
plt.ylim([0,1])
plt.grid(False)


plt.title('lithium H2O')

#n2b27
plt.subplot(4, 4, 4)
p1 = plt.bar(D, d, width, color='g')
p2 = plt.bar(D, f, width, bottom=d, color='b')
plt.ylim([0,1])
plt.grid(False)

plt.title('N2b27')

plt.show()        
      
