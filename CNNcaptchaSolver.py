# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 09:06:53 2025

@author: Breanna Sproul

ECCS4621.01 Deep Learning CAPCHA Assignment
Objective: In this assignment, you will implement a Convolutional Neural Network
(CNN) or a variant of it (such as CNN + LSTM, Attention-based CNN, etc.) to solve
CAPTCHA recognition problems. 
Using Kaggle dataset available in the Github repository

"""

#import needed libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

#get the dataset path and set up a way to access the image files
#if you download from our Github repo - unzip and make sure the extra sample folder is deleted before pasting your directory here
#data_dir = 'your directory here'
data_dir = 'C:/Users/brean/Documents/captcha_samples' #this is what mine is, but you will have to change it to your directory
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))] 

#set up the labels using dataset image filenames
#unrefined_labels = [img_path.split('the first part of your directory goes here')[-1] for img_path in image_files] #once again paste your directory up to the character right before the image label starts
unrefined_labels = [img_path.split('C:/Users/brean/Documents/captcha_samples\\')[-1] for img_path in image_files] #removes front part of string
labels = [img_path.split('.')[0] for img_path in unrefined_labels] #removes the last .png at the end to get only the char ID of image

#trying to process the characters better
characters = set(char for label in labels for char in label)
num_characters = sorted(list(characters))
print("Characters present: ", characters)
print("Number of unique characters: ", len(characters))

#OneHot Encoding to make labels that fit the model input requirement of numerical data instead of strings
label_size = 5
enc = tf.keras.layers.TextVectorization(split="character", output_mode="int")
enc.adapt(labels)
#print vocab to check that encoder is working
print(dict((str(v), k) for k,v in enumerate(enc.get_vocabulary())))
vocab_size = len(enc.get_vocabulary()) #original but gave 21 when unique is 19
#vocab_size = 19  #spyder didnt like this?
#print(f"Vocab_size = {vocab_size}")
print("Original Labels:")
print(labels[:2])
Y = enc(labels)
print(Y[:2])
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)
Y[:2]
#Y = Y.numpy() #for some reason Google Colab needed this line, but Spyder didnt. If your program throws a fit about no numpy array then uncomment this

#data splitting to training and testing data
train_images, test_images, train_labels, test_labels = train_test_split(image_files, Y, train_size = 0.85, test_size=0.15, random_state=0)

#load and preprocess images to be in numpy array
def load_and_preprocess_images(image_files):
  #Loads images from file paths, resizes and normalizes them
  images = []
  for img_path in image_files:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, (200, 50))  # Resize to match input shape, even tho most pics should already be at this level
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
print(f"Label: {train_labels[0]}")
print(f"Check if the original label only has 5 characters: {labels[0]}")

#look at the shape of the images and labels for correct model size adjustment
print(f"Image Shape: {test_images.shape}")
print(f"Label Shape: {test_labels.shape}")

#CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(50, 200, 3))) #changed from 16 to 32 for all CNN cov layers
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1100, activation='relu')) 
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5 * vocab_size, activation='softmax')) 
model.add(layers.Reshape((5, vocab_size)))

#extra summary for sizes of each layer when debugging
model.summary()

#compile and train model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) #keep lr at 0.001
history = model.fit(train_images, train_labels, epochs=60, validation_data=(test_images, test_labels))

#evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'lower right')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)


