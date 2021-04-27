import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
#from dfd import burst_observer
#from tf.keras.models import Sequential
#from tf.keras.layers import tf.keras.layers.Conv1D, tf.keras.layers.MaxPooling1D, tf.keras.layers.BatchNormalization
#from tf.keras.layers.core import tf.keras.layers.core.Activation, tf.keras.layers.core.Flatten, tf.keras.layers.core.Dense, tf.keras.layers.core.Dropout
#from tf.keras.layers.advanced_tf.keras.layers.core.Activations import tf.keras.activations.elu
#from tf.keras.initializers import tf.keras.initializers.glorot_uniform

class DFNet:
    @staticmethod
    def build(input_shape, classes):
        #print("This is the type")
        #print(type(input_shape.astype('float32')))
        model = tf.keras.models.Sequential()
        #Block1
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]

        model.add(tf.keras.layers.Conv1D(filters=filter_num[coun1], kernel_size=kernel_size[1], input_shape=input_shape,
                         strides=conv_stride_size[1], paddincoung='same',
                         name='block1_conv1',activation = 'elu'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        #model.add(tf.keras.activations.elu(input_shape.astype('float32'), alpha=1.0))#)#)#, name='block1_adv_act1'))
        model.add(tf.keras.layers.Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv2',activation = 'elu'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        #model.add(tf.keras.activations.elu(input_shape,alpha=1.0))#, name='block1_adv_act2'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block1_pool'))
        model.add(tf.keras.layers.Dropout(0.1, name='block1_Dropout'))

        model.add(tf.keras.layers.Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], paddincoung='same',
                         name='block2_conv1'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='block2_act1'))

        model.add(tf.keras.layers.Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv2'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='block2_act2'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool'))
        model.add(tf.keras.layers.Dropout(0.1, name='block2_Dropout'))

        model.add(tf.keras.layers.Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv1'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='block3_act1'))
        model.add(tf.keras.layers.Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv2'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='block3_act2'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool'))
        model.add(tf.keras.layers.Dropout(0.1, name='block3_Dropout'))

        model.add(tf.keras.layers.Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv1'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='block4_act1'))
        model.add(tf.keras.layers.Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv2'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='block4_act2'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                               padding='same', name='block4_pool'))
        model.add(tf.keras.layers.Dropout(0.1, name='block4_Dropout'))

        model.add(tf.keras.layers.Flatten(name='Flatten'))
        model.add(tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc1'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='fc1_act'))

        model.add(tf.keras.layers.Dropout(0.7, name='fc1_Dropout'))

        model.add(tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc2'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='fc2_act'))

        model.add(tf.keras.layers.Dropout(0.5, name='fc2_Dropout'))

        model.add(tf.keras.layers.Dense(classes, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc3'))
        model.add(tf.keras.layers.Activation('softmax', name="softmax"))
        return model
        
class DNNAttack:
    @staticmethod
    def build(input_shape, classes):

        model = tf.keras.models.Sequential()
        #Block1
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]
        # model.add(tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc1'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Activation('relu', name='fc1_act'))
        # model.add(tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc2'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Activation('relu', name='fc2_act'))
        # model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
        #                        padding='same', name='block1_pool'))
        # model.add(tf.keras.layers.Dropout(0.5, name='block1_Dropout'))
        # model.add(tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc3'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Activation('relu', name='fc3_act'))
        model.add(tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc4'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='fc4_act'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block2_pool'))
        model.add(tf.keras.layers.Dropout(0.5, name='block2_Dropout'))
        model.add(tf.keras.layers.Flatten(name='Flatten'))
        
        model.add(tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc5'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='fc5_act'))
        model.add(tf.keras.layers.Dropout(0.7, name='block3_Dropout'))
        model.add(tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc6'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu', name='fc6_act'))
        model.add(tf.keras.layers.Dropout(0.5, name='block4_Dropout'))
        model.add(tf.keras.layers.Dense(classes, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0), name='fc7'))
        model.add(tf.keras.layers.Activation('softmax', name="softmax"))
        return model

#from tensorflow import keras

import pickle
import numpy as np

def load_stuff():
 # Load training data
    #X_train = np.empty(shape = [1,1], dtype=object)
    with open('Datasets/ClosedWorld/NoDef/y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='latin1'))
    with open('completely_random_train_X_15k.npy', 'rb') as handle:
        #with open('Datasets/ClosedWorld/NoDef/X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(np.load(handle))
    #print("Made it this far")


        #print(count)
        #X_train = np.array(pickle.load(handle), dtype=object)

    # Load validation data
    with open('completely_random_valid_X_15k.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
  
    with open('Datasets/ClosedWorld/NoDef/y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle, encoding='latin1'))

    # Load testing data
    #with open('random_modified_test_X.pkl', 'rb') as handle:
        #X_test = np.array(pickle.load(handle))
        
    
    #with open('Datasets/ClosedWorld/NoDef/y_test_NoDef.pkl', 'rb') as handle:
        #y_test = np.array(pickle.load(handle, encoding='latin1'))

    print(len(X_train))
    #modified = burst_observer(X_test[0], .5, 1)
    #print(modified)
    #print(X_test[0])
    #modified = [str(x) for x in modified]
    #print(modified)
    #temp = ""
    #temp.join(modified)
    #print(type(temp))
    #f = open("demofile2.txt", "a")
    #f.write(temp)
# # This reads in a previous model
# json_file = open('model_35.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# print("Reading in weights from previous model")
# model.load_weights("random_values_model_35_epochs.h5")
# print("Compiling Previous model")
# model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
#     metrics=["accuracy"])
    #f.close()
    #f = open("demofile1.txt", "a")
    #temp = ""
    #what = [str(x) for x in X_test[0]]
    #print(what)
    #f.write(temp.join(what))
    #f.close()
    #for i in modified:
     #   print(i)
      #  for j in X_train[count]:
       #     print("Original")
        #    print(j)
        
# # This reads in a previous model
# json_file = open('model_35.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# print("Reading in weights from previous model")
# model.load_weights("random_values_model_35_epochs.h5")
# print("Compiling Previous model")
# model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
#     metrics=["accuracy"])
        
# # This reads in a previous model
# json_file = open('model_35.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)coun
# # load weights into new model
# print("Reading in weights from previous model")
# model.load_weights("random_values_model_35_epochs.h5")
# print("Compiling Previous model")
# model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
#     metrics=["accuracy"])
        
# # This reads in a previous model
# json_file = open('model_35.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# print("Reading in weights from previous model")
# model.load_weights("random_values_model_35_epochs.h5")
# print("Compiling Previous model")
# model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
#     metrics=["accuracy"])
        #count = count + 1
    
    #print("X: Training data : ", X_train)
    #print("y: Training data", y_train)
    #print("X: Validation data ", X_valid)
    #print("y: Validation data ", y_valid)
    #print("X: Testing data", X_test)
    #print("y: Testing data", y_test)
    #print("Test: ",X_test)
    #print("Data dimensions:")
    ##print("X: Training data's shape : ", X_train.shape)
    #print("y: Training data's shape : ", y_train.shape)
    #print("X: Validation data's shape : ", X_valid.shape)
    #print("y: Validation data's shape : ", y_valid.shape)
    #print("X: Testing data's shape : ", X_test.shape)
    #print("y: Testing data's shape : ", y_test.shape)
    #print(X_train.isnull().any())
    #print(y_train.isnull().any())
    #print(X_valid.isnull().any())
    #print(y_valid.isnull().any())
    return X_train, y_train, X_valid, y_valid#, X_test, y_test
    

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
NB_EPOCH = 5   # Number of training epoch
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE =  16# Batch size
VERBOSE = 2 # Output display mode
LENGTH = 15000 # Packet sequence length
OPTIMIZER = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

NB_CLASSES = 95 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid = load_stuff()#, X_test, y_test = load_stuff()
# Please refer to the dataset format in readme
tf.keras.backend.set_image_data_format("channels_last") # tf is tensorflow

# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
#X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
#y_test = y_test.astype('float32')

# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]#json_file = open('model.json', 'r')
#X_test = X_test[:, :,np.newaxis]#loaded_model_json = json_file.read()
#json_file.close()
print(X_train.shape[0], 'train sampl#model = model_from_json(loaded_model_json)es')
print(X_valid.shape[0], 'validation samples')
#print(X_test.shape[0], 'test samples')

# Convert class vectors to categorical classes matrices
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_valid = tf.keras.utils.to_categorical(y_valid, NB_CLASSES)
#y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

# Building and training model
print("Building and training DF model")

#### This is to start a model ##########################################
model = DNNAttack.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
     metrics=["accuracy"])
print("Model compiled")
###########################################################################
count = 0

history = model.fit(X_train, y_train,
            batch_size=BATCH_SIZE, epochs=NB_EPOCH,
            verbose=VERBOSE, validation_data=(X_valid, y_valid))

count = count + NB_EPOCH
filename = 'model_' + str(count) + '_soverfitting.json'
model_name = 'completely_random_values_model_20k_' + str(count) + '_epochs_no_overfitting.h5'
model_json = model.to_json()
with open(filename, "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
model.save_weights(model_name)
print("Saved model to disk")
# # This reads in a previous model
# while count < 100 :
    
#     filename = 'model_' + str(count) + '_soverfitting.json'
#     json_file = open(filename, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     # load weights into new model
#     print("Reading in weights from previous model")
#     model_name = 'completely_random_values_model_' + str(count) + '_epochs_no_overfitting.h5'
#     model.load_weights(model_name)
#     print("Compiling Previous model")
#     model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
#         metrics=["accuracy"])

#     print("Continuing training on previous model")
#     # Start training
#     history = model.fit(X_train, y_train,
#             batch_size=BATCH_SIZE, epochs=NB_EPOCH,
#             verbose=VERBOSE, validation_data=(X_valid, y_valid))

#     count = count + NB_EPOCH
#     filename = 'model_' + str(count) + '_soverfitting.json'
#     model_name = 'completely_random_values_model_' + str(count) + '_epochs_no_overfitting.h5'
#     model_json = model.to_json()
#     with open(filename, "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights(model_name)
#     print("Saved model to disk")

#load_stuff()
