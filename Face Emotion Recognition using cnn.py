#!/usr/bin/env python
# coding: utf-8

# # Face Emotion Detection Using Convolutional Neural Network (CNN)

# ## Importing the necessary libraries and plotting the sample images from all the 7 classes in the training directory

# In[1]:


import numpy as np
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

# size of the image: 48*48 pixels
pic_size = 48

# input path for the images
base_path = "C:/Users/dvgna/Desktop/FER/images/"

plt.figure(0, figsize=(12,20))
cpt = 0

for expression in os.listdir(base_path + "train/"):
    for i in range(1,6):
        cpt = cpt + 1
        plt.subplot(7,5,cpt)
        img = load_img(base_path + "train/" + expression + "/" +os.listdir(base_path + "train/" + expression)[i], target_size=(pic_size, pic_size))
        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show()


# ## Counts of all the images in 7 classes in the training directory

# In[2]:


for expression in os.listdir(base_path + "train"):
    print(str(len(os.listdir(base_path + "train/" + expression))) + " " + expression + " images")


# ## Defining the data generator class for training and validation

# In[3]:


from keras.preprocessing.image import ImageDataGenerator

# number of images to feed into the NN for every batch
batch_size = 128

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

train_generator = datagen_train.flow_from_directory(base_path + "train",
                                                    target_size=(pic_size,pic_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = datagen_validation.flow_from_directory(base_path + "validation",
                                                    target_size=(pic_size,pic_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)


# In[12]:


filenames = validation_generator.filenames
nb_samples = len(filenames)
print(nb_samples)


# ## Defining the model's architecture with 4 CNN layers, 1 flatten and 2 fully connected and a output layer with 'Adam' as the optimiser and 'Categorical_Crossentropy' as the loss function with 'Accuracy' as the metric

# In[4]:


from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

# number of possible label values
nb_classes = 7

# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# ## Only Model's best performance is checkpointed and saved to model's weight with 50 epochs, the model's training took approx 23 hours and 15 mins !

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# number of epochs to train the NN\nepochs = 60\n\nfrom keras.callbacks import ModelCheckpoint\n\n\ncheckpoint = ModelCheckpoint("face_expression_model_weights.h5", monitor=\'val_acc\', verbose=1, save_best_only=True, mode=\'max\')\ncallbacks_list = [checkpoint]\nmodel.fit_generator(generator=train_generator,\n                                steps_per_epoch=train_generator.n//train_generator.batch_size,\n                                epochs=epochs,\n                                validation_data = validation_generator,\n                                validation_steps = validation_generator.n//validation_generator.batch_size,\n                                callbacks=callbacks_list\n                                )')


# In[6]:


print(model.history)


# ## Plotting the Training and Validation (Loss and Accuracy)

# In[15]:


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(model.history.history['loss'], label='Training Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(model.history.history['acc'], label='Training Accuracy')
plt.plot(model.history.history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


# ## Finally computing the predictions and plotting the actual vs predicted labels in a confusion matrix

# In[17]:


# compute predictions
predictions = model.predict_generator(generator=validation_generator, steps= nb_samples/batch_size)
y_pred = [np.argmax(probas) for probas in predictions]
y_test = validation_generator.classes
class_names = validation_generator.class_indices.keys()

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Normalized confusion matrix')
plt.show()

