# CNN_CAPTCHA_Solver
This is a project for ECCS4621.01 Deep Learning CAPTCHA Solver with CNNs Assignment. I have used a Convolutional Neural Network(CNN) model to break the CAPTCHAs found in the dataset from Wilhelmy, Rodrigo & Rosas, Horacio. (2013) obtained through [Kaggle](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images?resource=download). The CAPTCHAs in this dataset are all 200 x 50 grayscale images that consist of 5 characters(letters and numbers) with a blur and a line striking through them. This model outputs an accuracy of 75-85 depending on the way you run it. Google Colab generally gives lower estimates, while Python options installed on a computer(like Spyder) will provide a higher estimate. 

A Google Colab version of this code can be found [here](https://colab.research.google.com/drive/1CrxZCsEU87U5SiCA3VfOJkrnho8ltfFw#scrollTo=5PmL4CuOA5Q7) if Colab/ipynb format is preferred. The code is mostly the same however, it is recommended to follow the short README at the top to ensure files are uploaded correctly.  

# Required Libraries
This program mainly utilizes Tensorflow and Keras to create the CNN model. The dataset was prepared using os, numpy, cv2, and sklearn test_train_split libraries. Lastly matplotlib is used to evaluate the model at the end.
{put a pic of the code here}


```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
```

# Loading Dataset
You will need to download the dataset before continuing. If you obtain the dataset through this Github Repository, then you will need to unzip the file and delete the extra sample folder to avoid errors in the code. You will need to change some of the code to fit with your dataset directory. 

The first change is in the data_dir variable. This will simply require you to replace C:/Users/brean/Documents/captcha_samples with the directory of your downloaded dataset. 
```python
data_dir = 'C:/Users/brean/Documents/captcha_samples' #this is what mine is, but you will have to change it to your directory
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
```
The second change is when the labels for the images are being set up. The labels are made by splitting the filename until only the characters in the image are shown. You will need to make sure unrefined_labels has all of the filename before the 5 characters needed for the label. Use the print statement to make sure your labels are being created correctly.  
```python
unrefined_labels = [img_path.split('C:/Users/brean/Documents/captcha_samples\\')[-1] for img_path in image_files] #removes front part of string
labels = [img_path.split('.')[0] for img_path in unrefined_labels] #removes the last .png at the end to get only the char ID of image
...
print(f"Check if the original label only has 5 characters: {labels[0]}")
```

# Preparing Dataset
OneHot Encoding was used to convert the labels from String data to Numerical data which is required for the model to work. 
```python
enc = tf.keras.layers.TextVectorization(split="character", output_mode="int")
enc.adapt(labels)
Y = enc(labels)
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)
```
The dataset images were separated in an 85:15 format by sklearn train_test_split.
```python
train_images, test_images, train_labels, test_labels = train_test_split(image_files, Y, train_size = 0.85, test_size=0.15, random_state=0)
```
Lastly, the train and test images were processed to ensure each was normalized and placed in a numpy array.
```python
def load_and_preprocess_images(image_files):
  images = []
  for img_path in image_files:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, (200, 50))  # Resize to match input shape, even tho most pics should already be at this level
    img = img.astype('float32') / 255.0  # Normalize pixel values
    images.append(img)
  return np.array(images)
train_images = load_and_preprocess_images(train_images)
test_images = load_and_preprocess_images(test_images)
```

# CNN Model
The CNN model uses Conv2D and MaxPooling2D layers with ReLU activation and varying numbers of neuron layers. Two Dense layers are found at the end of the model, with the output layer using SoftMax activation. A Reshape was performed at the end to allow the model to be compiled. Dropout layers are added throughout the model to increase test accuracy and decrease overfitting. Finally, the model was compiled with Adam optimizer and categorical_crossentropy. 

The last part of the code evaluates the model and creates a plot of the history of the model's accuracy.

