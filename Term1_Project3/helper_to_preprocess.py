import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2, math

import h5py

from datetime import datetime

def strip_date_time(string):
	return datetime.strptime(string.split("/")[-1],"center_%Y_%m_%d_%H_%M_%S_%f.jpg")

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image,steer,trans_range):
    rows,cols = image.shape[:-1]

    #Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y)-(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
            image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image


new_size_col,new_size_row = 64, 64

def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)    
    #image = image/255.-.5
    return image

def preprocess_image_file_train(line_data):
    #i_lrc = np.random.randint(3)
    i_lrc = 1
    if (i_lrc == 0):
        path_file = line_data['left'].tolist()[0].strip()
        shift_ang = .25
    elif (i_lrc == 1):
        path_file = line_data['center'].tolist()[0].strip()
        shift_ang = 0.
    else:
        path_file = line_data['right'].tolist()[0].strip()
        shift_ang = -.25
    
    y_steer = line_data['steering'].tolist()[0] + shift_ang
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	
    if np.random.randint(2)==0:
        image,y_steer = trans_image(image,y_steer,100)
    else:
        image = augment_brightness_camera_images(image)
    #image = preprocessImage(image)
    image = np.array(image)
    	
    if np.random.randint(2)==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    	
    return image,y_steer

def helper_to_balance_dataset(camera_to_save="center"):
    #first create features and labels
    df1 = pd.read_csv("data4/driving_log.csv",names=["center","left","right","steering","throttle","break","speed"])
    df2 = pd.read_csv("data3/driving_log.csv",names=["center","left","right","steering","throttle","break","speed"])
    df4 = pd.read_csv("data2/driving_log.csv",names=["center","left","right","steering","throttle","break","speed"])

    #get shape of single image
    img_shape = list(cv2.imread(df1[camera_to_save][0]).shape)

    data = df1.append(df2)
    data = data.append(df4)

    data.reset_index(drop=True, inplace=True)
    print(data.shape)	
    #first remove skewed part
    chop_data = data[~(abs(data["steering"])<0.02)].copy()

    #count histograms 
    patch = plt.hist(chop_data["steering"],bins=100)
    bin_count, bins = patch[0].astype(int),patch[1]
    groups = chop_data.groupby(pd.cut(chop_data["steering"],bins))

    print(bin_count.max())

    #number of times to augment data
    frac_augment = bin_count.max()//(bin_count+1)
    f = h5py.File("driving_feature_balanced.h5","w")
    
    cnt = 0
    for name,group in groups:
        chop_data.ix[group.index,"fraction"] = frac_augment[cnt]
        cnt+=1
    
    global_count = chop_data["fraction"].sum().astype(int)+3*bin_count.max()

    #create group names
    features_dataset = f.create_dataset("features",shape=[global_count]+img_shape,dtype=np.uint8)
    labels_dataset = f.create_dataset("labels",shape=[global_count],dtype=np.float32)

    index = 1
    central_data = data[abs(data["steering"])<0.02]
    #now add the rest of central data
    for j_line in central_data.index:
        image = cv2.imread(central_data.ix[j_line,"center"])
        features_dataset[index,:,:,:] = image
        labels_dataset[index] = central_data.ix[j_line,"steering"] 
        index+=1

        if index>bin_count.max()*2:
            print("Breaking loop ",index)
            break

    for i_line in chop_data.index:
        #print(i_line)
        i_augment = chop_data.ix[i_line,"fraction"].astype(int)
        line_data = chop_data.ix[[i_line]]

        for i in range(i_augment):
            x,y = preprocess_image_file_train(line_data)    
            features_dataset[index,:,:,:] = x
            labels_dataset[index] = y
            index+=1

        if index%10==0: 
            print("Processing image {0:4d}, of {1:4d}".format(index,i_line))

    f.close() 

if __name__=="__main__":
    helper_to_balance_dataset(camera_to_save="center")
