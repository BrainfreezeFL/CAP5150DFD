import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

def load_stuff():
 # Load training data
    # with open('drive/MyDrive/ClosedWorld/NoDef/y_train_NoDef.pkl', 'rb') as handle:
    #     y_train = np.array(pickle.load(handle, encoding='latin1'))
    # with open('drive/MyDrive/ClosedWorld/NoDef/X_train_NoDef.pkl', 'rb') as handle:
    #     X_train = np.array(pickle.load(handle, encoding='latin1'))

    # # Load validation data
    # with open('drive/MyDrive/ClosedWorld/NoDef/X_valid_NoDef.pkl', 'rb') as handle:
    #     X_valid = np.array(pickle.load(handle, encoding='latin1'))
    # with open('drive/MyDrive/ClosedWorld/NoDef/y_valid_NoDef.pkl', 'rb') as handle:
    #     y_valid = np.array(pickle.load(handle, encoding='latin1'))

    # # Load testing data
    # with open('drive/MyDrive/ClosedWorld/NoDef/X_test_NoDef.pkl', 'rb') as handle:
    #     X_test = np.array(pickle.load(handle, encoding='latin1'))
    # with open('drive/MyDrive/ClosedWorld/NoDef/y_test_NoDef.pkl', 'rb') as handle:
    #     y_test = np.array(pickle.load(handle, encoding='latin1'))
    # print("Data dimensions:")
    # print("X: Training data's shape : ", X_train.shape)
    # print("y: Training data's shape : ", y_train.shape)
    # print("X: Validation data's shape : ", X_valid.shape)
    # print("y: Validation data's shape : ", y_valid.shape)
    # print("X: Testing data's shape : ", X_test.shape)
    # print("y: Testing data's shape : ", y_test.shape)
    # with open('Datasets/ClosedWorld/NoDef/y_train_NoDef.pkl', 'rb') as handle:
    #     y_train = np.array(pickle.load(handle, encoding='latin1'))
    # with open('random_modified_train_X.npy', 'rb') as handle:
    #     #with open('Datasets/ClosedWorld/NoDef/X_train_NoDef.pkl', 'rb') as handle:
    #     X_train = np.array(np.load(handle))
    # #print("Made it this far")
    with open('Datasets/ClosedWorld/NoDef/y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='latin1'))
    with open('random_modified_train_X.npy', 'rb') as handle:
        #with open('Datasets/ClosedWorld/NoDef/X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(np.load(handle))
    #print("Made it this far")


        #print(count)
        #X_train = np.array(pickle.load(handle), dtype=object)

    # Load validation data
    with open('random_modified_valid_X.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
  
    with open('Datasets/ClosedWorld/NoDef/y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle, encoding='latin1'))

    # Load testing data
    with open('random_modified_test_X.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
        
    
    with open('Datasets/ClosedWorld/NoDef/y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle, encoding='latin1'))


        #print(count)
        #X_train = np.array(pickle.load(handle), dtype=object)

    # Load validation data
    with open('random_modified_valid_X.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
  
    with open('Datasets/ClosedWorld/NoDef/y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle, encoding='latin1'))

    # Load testing data
    #with open('random_modified_test_X.pkl', 'rb') as handle:
        #X_test = np.array(pickle.load(handle))
        
    
    #with open('Datasets/ClosedWorld/NoDef/y_test_NoDef.pkl', 'rb') as handle:
        #y_test = np.array(pickle.load(handle, encoding='latin1'))

    return X_train, y_train, X_valid, y_valid, X_test, y_test
    

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is to implement deep fingerprinting model for website fingerprinting attacks
# ACM Reference Formant
# Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018.
# Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.
# In 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS ’18),
# October 15–19, 2018, Toronto, ON, Canada. ACM, New York, NY, USA, 16 pages.
# https://doi.org/10.1145/3243734.3243768


#from keras import backend as K
#from utility import LoadDataNoDefCW
#from Model_NoDef import DFNet
import random
#from keras.utils import np_utils
#from keras.optimizers import Adamax
import numpy as np
import os

random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Use only CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

description = "Training and evaluating DF model for closed-world scenario on non-defended dataset"

print(description)
# Training the DF model
NB_EPOCH = 8   # Number of training epoch
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 128 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 15000 # Packet sequence length
OPTIMIZER = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

NB_CLASSES = 95 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = load_stuff()
# Please refer to the dataset format in readme
tf.keras.backend.set_image_data_format("channels_last") # tf is tensorflow

# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to categorical classes matrices
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_valid = tf.keras.utils.to_categorical(y_valid, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

#json_file = open('model_31_soverfitting.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("random_values_model_31_epochs_no_overfitting.h5")
#model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
#    metrics=["accuracy"])
#print("Loaded model from disk")
#score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
#print("Original Model Testing accuracy:", score_test[1])

json_file = open('vulnerability_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("vulnerability_model_weights.h5")
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
    metrics=["accuracy"])
print("Loaded model from disk")
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Attempted Fix Model Testing accuracy:", score_test[1])

