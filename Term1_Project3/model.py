import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2, math
from keras import backend as K
import pickle as pickle
import tensorflow as tf
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Activation, ELU
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Lambda, Input
from keras.models import Model, Sequential
from keras import optimizers

import h5py


new_size_col,new_size_row = 64, 64

def preprocessImage(image):
	shape = image.shape
	# note: numpy arrays are (row, col)!
	image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
	image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)    
	#image = image/255.-.5
	return image


def load_h5_file(file_name="mytestfile.h5"):
	f = h5py.File(file_name,"r")

	#right now returning everything, if memory doesn't allow do something smarter
	X,y = f["features"],f["labels"]
	
	return X,y

def plot_layers(X):

    #pick one of imgaes
    image = X[1000]

    #load model
    model = load_model("model.h5",custom_objects={"tf":tf,"new_size_col":new_size_col,"new_size_row":new_size_row})

    #summary of model
    print(model.summary())	

    #pick convolution layer
    get_activation = K.function([model.layers[0].input],[model.layers[3].output])
    output = get_activation([image[np.newaxis,:,:,:]])[0]

    #get only first image
    print(output.shape)

    fig,axes = plt.subplots(ncols=5,nrows=4,figsize=(15,12))
    ax = axes.ravel()

    for i in range(20):
        ax[i].imshow(output[0,:,:,i],cmap="gray")

    plt.tight_layout()
    plt.show()

def dataset_generator(X,y,batch_size,train=True,train_fraction=0.65):
    n_examples = X.shape[0]
    train_size = int(X.shape[0]*train_fraction)
    test_size = n_examples-train_size
    train_idx = np.random.choice([i for i in range(n_examples)],train_size,replace=False)	
    test_idx = np.array([i for i in range(n_examples) if i not in train_idx])	
    train_idx.sort()
    test_idx.sort()

    while 1:
        if train: #choose training dataset
            for i in range(0,train_size,batch_size):
                mask = np.zeros(n_examples,dtype=bool)
                mask[train_idx[i:i+batch_size]] = True
                yield X[mask,:], y[mask]
        else:
            for j in range(0,test_size,batch_size):
                mask = np.zeros(n_examples,dtype=bool)
                mask[test_idx[j:j+batch_size]] = True
                yield X[mask,:], y[mask]

def convolution_model(X,y,saved_model="model.h5"):
    N,H,W,C = X.shape
    print(X.shape)

    #create model
    model = Sequential()
    model.add(Lambda(lambda image: tf.image.crop_to_bounding_box(
        image, offset_height=60, offset_width=0, target_height=100, target_width=320), input_shape=(H,W,C)))
	
    model.add(Lambda(lambda image: image/255.0 - 0.5))
	
    model.add(Convolution2D(24,(5,5),padding="valid"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D((2,2),padding="valid"))
    model.add(Dropout(0.3))
    model.add(Convolution2D(36,(5,5),padding="valid"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D((2,2),padding="valid"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    #check if model already exist
    model = load_model(saved_model,custom_objects={"tf":tf,"new_size_col":new_size_col,"new_size_row":new_size_row})

    optimizer = optimizers.Adam(lr=0.0001, decay=1e-5)
    model.compile(loss="mse",optimizer=optimizer)
	
    BATCH_SIZE = 64
    TRAIN_FRACTION = 0.65
    hist = model.fit_generator(
        dataset_generator(X,y,batch_size=BATCH_SIZE,train_fraction=TRAIN_FRACTION,train=True), 
        steps_per_epoch = int(TRAIN_FRACTION*X.shape[0]/BATCH_SIZE), 
        validation_data = dataset_generator(X,y,batch_size=BATCH_SIZE,train_fraction=TRAIN_FRACTION,train=False),
        validation_steps = int((1-TRAIN_FRACTION)*X.shape[0]/BATCH_SIZE),  
        class_weight = None, nb_worker=1, nb_epoch = 10, verbose=1, callbacks=[])
	
    #save model
    model.save("model.h5")	

if __name__=="__main__":
    X,y = load_h5_file(file_name="driving_feature_balanced.h5")
    #plot_layers(X)
    convolution_model(X,y) 
