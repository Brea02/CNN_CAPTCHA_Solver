# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 09:06:53 2025

@author: Breanna Sproul

ECCS4621.01 Deep Learning CAPCHA Assignment
Objective: In this assignment, you will implement a Convolutional Neural Network
(CNN) or a variant of it (such as CNN + LSTM, Attention-based CNN, etc.) to solve
CAPTCHA recognition problems. 
Using Kaggle dataset available in the Github repository

Currently:
    can access the images from dataset and made labels for it
    shows an image and its corresponding label
    model is not accepting the labels so might have not done labels right{might fix itself when separating char}
Needs:
    get the model to accept labels and run dataset
    add processing step to separate each image into individual character components
    Achieve approximately 80% accuracy on the test set -- listed in assignment
    rewrite some code to make it easier for other computers to use
"""

#import needed libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 
from sklearn.model_selection import train_test_split

#get the dataset path and set up a way to access images
#data_dir = 'your directory here'
data_dir = 'C:/Users/brean/Documents/captcha_samples'
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]


#set up the labels using dataset image filenames
#!! I might have done this wrong and need a different type of labels for the model to work
unrefined_labels = [img_path.split('C:/Users/brean/Documents/captcha_samples\\')[-1] for img_path in image_files]
labels = [img_path.split('.')[0] for img_path in unrefined_labels]

#data splitting
train_images, test_images, train_labels, test_labels = train_test_split(image_files, labels, train_size = 0.85, test_size=0.15, random_state=0)

#load and preprocess images to be in numpy array
def load_and_preprocess_images(image_files):
  """Loads images from file paths, resizes and normalizes them."""
  images = []
  for img_path in image_files:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    #img = cv2.resize(img, (32, 32))  # Resize to match input shape, but looks really bad
    img = img.astype('float32') / 255.0  # Normalize pixel values
    images.append(img)
  return np.array(images)
#use ^ to turn image file path to image data in a numpy array
train_images = load_and_preprocess_images(train_images)
test_images = load_and_preprocess_images(test_images)

#look at one of the pictures to make sure everything worked
#img_path = train_images[0]
#img = cv2.imread(img_path)  # Load image using OpenCV
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
#img = img.astype('float32') / 255.0 # Normalize pixel values
#plt.imshow(cv2.imread(train_images[0]))
plt.imshow(train_images[0])
plt.axis('off')
plt.show()
print(train_labels[0])



'''
#create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#compile and train the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

#evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'lower right')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
'''

