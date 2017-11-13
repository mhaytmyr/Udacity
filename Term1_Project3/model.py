import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Activation, ELU
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Lambda, Input
from keras.models import Model, Sequential
from keras import optimizers

import h5py



#-----------------------------------------------------
#Helper function to save frames to hiearchical data file
#-----------------------------------------------------
def save_feature_hd5(camera_to_save="center"):
        #first create features and labels
        data = pd.read_csv("data/driving_log.csv",names=["center","left","right","steering","throttle","break","speed"])

        #get shape of single image
        img_shape = list(cv2.imread(data[camera_to_save][0]).shape)

        data.reset_index(drop=True, inplace=True)
        print(data.shape)

        #create hierarchical data format (hdf5) file
        f = h5py.File("driving_feature_map.h5","w")

        #create group names
        n_augment = 2
        features_dataset = f.create_dataset("features",shape=[data.shape[0]*n_augment]+img_shape,dtype=np.uint8)
        labels_dataset = f.create_dataset("labels",shape=[data.shape[0]*n_augment],dtype=np.float32)
        
        for i in range(0,data.shape[0]):
                
                #create images and labels
                img1 = cv2.imread(data[camera_to_save][i])

                #augment data by flipping it w.r.t y-coordinate
                img2 = cv2.flip(img1,1)
                images = np.concatenate([img1[np.newaxis],img2[np.newaxis]],axis=0)
                
                label1 = data["steering"][i]
                label2 = -1.0*label1
                labels = np.array([label1,label2]) 

                #save images to hdf
                features_dataset[i*n_augment:(i+1)*n_augment,:,:,:] = images
                labels_dataset[i*n_augment:(i+1)*n_augment] = labels
        
                if i%500==0: print("Processing image {0:4d}".format(i))

        f.close() 


#----------------------------------------
#Helper function to load features dataset
#----------------------------------------
def load_h5_file(file_name="mytestfile.h5"):
        f = h5py.File(file_name,"r")

        #right now returning everything, if memory doesn't allow do something smarter
        X,y = f["features"],f["labels"]

        return X[:],y[:]

#-----------------------------------------------
#Helper function to create model and train model
#-----------------------------------------------
def convolution_model(X,y,saved_model="model.h5"):
        N,H,W,C = X.shape

        model = Sequential()
	#pre-process image, chop-off the top 60 pixels of image
        model.add(Lambda(lambda image: tf.image.crop_to_bounding_box(
                image, offset_height=60, offset_width=0, target_height=100, target_width=320), input_shape=(H,W,C)))

        model.add(Lambda(lambda image: image/255.0 - 0.5))
        model.add(Convolution2D(50,(5,5),padding="valid"))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((2,2),padding="valid"))
        model.add(Dropout(0.3))
        model.add(Convolution2D(100,(5,5),padding="valid"))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((2,2),padding="valid"))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Dropout(0.3))
        model.add(Dense(1))

        #check if model already exist
        #model = load_model(saved_model,custom_objects={"tf":tf})

        sgd = optimizers.SGD(lr=0.0001, decay=1e-5, momentum=0.95, nesterov=True)
        model.compile(loss="mse",optimizer=sgd)
        hist = model.fit(X.astype(np.float32),y.astype(np.float32),batch_size=64,
                validation_split=0.35, shuffle=True, epochs=300, verbose=1)

	#save model
        model.save("model.h5")

        #plot validation and training loss
        fig, axes = plt.subplots(ncols=2,figsize=(12,8))
        ax = axes.ravel()
        ax[0].plot(hist.history["loss"],"r-.",label="Train loss")
        ax[0].legend()
        ax[1].plot(hist.history["val_loss"],"g--",label="Validation loss")
        ax[1].legend()
        plt.show()

#------------------------------
#Main function to run all steps
#------------------------------
if __name__=="__main__":
        X,y = load_h5_file(file_name="driving_feature_small.h5")
        convolution_model(X,y)











	
