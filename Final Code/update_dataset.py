#from tensorflow import keras

import pickle
import numpy as np

import random
import tensorflow as tf
from tensorflow import keras


import math

def inject(input, rate, what_to_inject, position, prev_buffer_size, randoms):
    
    if (randoms == False) : 
        array = [None,None]
        injection_amount = rate * prev_buffer_size
        injection_amount = math.ceil(injection_amount)
        x = 0
        while x < injection_amount :
        
            input.insert(position, what_to_inject)
            x = x + 1
            
        array[0] = input
        array[1] = injection_amount
        return array
    else :
        array = [None,None]
        injection_amount = rate
        injection_amount = math.ceil(injection_amount)
        x = 0
        while x < injection_amount :
        
            input.insert(position, what_to_inject)
            x = x + 1
            
        array[0] = input
        array[1] = injection_amount
        return array



# This can also be considered one way injection. Injection method of 1 is client side and -1 is server side. 
def burst_observer(input, perturbation_rate, injection_method, randoms):
    random.seed(0)
    output = list(input)
    previous_buffer_length = 2
    current_buffer_length = 0
    injection_buffer = injection_method
    pos_input = 0
    pos_output = 0
    array = [None,None]
    for i in input:
        if (randoms == True):
            #print("Changed")
            perturbation_rate = random.randint(0,8)
        if pos_input > 0 :
            if i == injection_method:
                if i == input[pos_input-1] and current_buffer_length == 1:
                    array = inject(output, perturbation_rate, injection_buffer, pos_output, previous_buffer_length, random)
                    pos_output = array[1] + pos_output
                    output = array[0]
                    current_buffer_length = current_buffer_length + 1
                elif i == input[pos_input-1] and current_buffer_length != 1:
                    current_buffer_length = current_buffer_length + 1
                pos_input = pos_input + 1
                pos_output = pos_output + 1
                
            else :
                if input[pos_input-1] == injection_method:
                    previous_buffer_length = current_buffer_length
                current_buffer_length = 0
                pos_input = pos_input + 1
                pos_output = pos_output + 1
        elif i == injection_method:
            
            current_buffer_length = current_buffer_length + 1
            pos_input = pos_input + 1
            pos_output = pos_output + 1
        
        else : 
            current_buffer_length = 0
            pos_input = pos_input + 1
            pos_output = pos_output + 1
    #print(output)
    return output
            


# This performs two way injection by simply running the one way injection algorithm on the same string.
def two_way_injection(input, client_rate, server_rate, randoms) :
    #print(input)
    output = burst_observer(input, client_rate, 1, randoms)
    #print(output)
    output = burst_observer(output, server_rate, -1, randoms)
    #print("#######################################################################")
   # print(output)
    while (len(output) < 15000):
        output.append('0')
    return output

def load_stuff():
 # Load training data
    
    #with open('Datasets/ClosedWorld/NoDef/y_train_NoDef.pkl', 'rb') as handle:
    #    y_train = np.array(pickle.load(handle, encoding='latin1'))
    #with open('Datasets/ClosedWorld/NoDef/X_train_NoDef.pkl', 'rb') as handle:
        #X_train = np.array(pickle.load(handle, encoding='latin1'))

    # Load validation data
    with open('Datasets/ClosedWorld/NoDef/X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle, encoding='latin1'))
    #with open('Datasets/ClosedWorld/NoDef/y_valid_NoDef.pkl', 'rb') as handle:
    #    y_valid = np.array(pickle.load(handle, encoding='latin1'))

    # Load testing data
    #with open('Datasets/ClosedWorld/NoDef/X_test_NoDef.pkl', 'rb') as handle:
        #X_test = np.array(pickle.load(handle, encoding='latin1'))
    #with open('Datasets/ClosedWorld/NoDef/y_test_NoDef.pkl', 'rb') as handle:
    #    y_test = np.array(pickle.load(handle, encoding='latin1'))
    #count = 0
    
    #for i in modified:
     #   print(i)
      #  for j in X_train[count]:
       #     print("Original")
        #    print(j)
         #   print("Modified")
         #   print(modified[count])
        
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
    
    return X_valid#X_train, y_train, X_valid, y_valid, X_test, y_test
    

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
#import random
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

#print(description)
# Training the DF model
NB_EPOCH = 3   # Number of training epoch
#print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 128 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 15000 # Packet sequence length
OPTIMIZER = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

NB_CLASSES = 95 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print("Updating Dataset")
X_valid = load_stuff()#y_train, X_valid, y_valid, X_test, y_test = load_stuff()
# Please refer to the dataset format in readme

first_counter = 0


limit = len(X_valid)
updated_data = np.zeros(shape = [limit, 15000])
count = 0
print(limit)
print("Getting test")

while count < limit :

    items = X_valid[count]
    #print("Original")
    #print(items[:30])
    client = random.random()
    #print(client)
    server = random.random()
        #print("Before")
    modified = two_way_injection(items, client, server, True)
        #print("After")
    #modified = burst_observer(X_valid[count], .5, 1)
    #print("Modified by Algorithm")
    #print(modified[:30])
    #print(X_test[0])
    #modified2 = [str(x) for x in modified]
    #print("Double Checking No changes")
    #print(modified2[:30])
    temp = ""
    #temp.join(modified2)
    #print(type(temp))
    #f = open("demofile2.txt", "a")
    #f.write(temp.join(modified2))
    #f.close()
    #f = open("demofile1.txt", "a")
    #temp2 = ""
    #what = [str(x) for x in items]
    #print(what)
    #f.write(temp2.join(what))
    #f.close()
    
    updated_data[count] = modified
        #print("Dump")
        
    #firs_counter = firt_counter + 1
        #print("Count")
    if count % 1000 == 0 :
        print(math.ceil(count/limit*100), "% Complete", end="\r")
    count = count + 1
    
print(len(updated_data))    
print("Writing to File")
#print(updated_data)
with open('completely_random_valid_X_15k.pkl', 'wb') as f:
    pickle.dump(updated_data,f, pickle.HIGHEST_PROTOCOL)
    
    
#with open('completely_random_train_X_15k.npy', 'wb') as f:
    #np.save(f, updated_data)
    
print("Finished Writing to File")

#print("Now opening file back up")
#with open('completely_random_valid_X.pkl', 'rb') as handle:
        #test = np.array(pickle.load(handle, encoding='latin1'))
#print(updated_data[0][:30])   
#print(test[0][:30])
