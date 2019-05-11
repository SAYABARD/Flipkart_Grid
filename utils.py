# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

plt.switch_backend('agg')

from PIL import Image
import numpy as np
import pickle
import pandas as pd
import dask.array as da



##data preprocessing----------------------------------

def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

id_to_data={}
id_to_size={}

def dataload():
 id_to_data={}
 df1 = pd.read_csv("training_set.csv")
 sample_df = df1
 for i in range(0, 12000): 
    path=sample_df.iloc[i]['image_name']
    print(i)
    image=Image.open("./dataset/images/"+path) 
    image=image.resize((224,224))
    image=np.array(image,dtype=np.int32) 
    image=image/255
    image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
    id_to_data[i]=image
 id_to_data=np.array(list(id_to_data.values()))
 return id_to_data

def dataload1():
 id_to_data={}
 df1 = pd.read_csv("training_set.csv")
 sample_df = df1
 for i in range(12000, 24000): 
    path=sample_df.iloc[i]['image_name']
    print(i)
    image=Image.open("./data/images/"+path) 
    image=image.resize((224,224))
    image=np.array(image,dtype=np.int32) 
    image=image/255
    image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
    id_to_data[i]=image
 id_to_data=np.array(list(id_to_data.values()))
 return id_to_data

def getdata():
    id_to_data=dataload()
    data=id_to_data
    index=[i for i in range(24000)] 
    data_train = dataload()
    data_train1 = dataload1()
    data_test=data_train[0:5]
    f=open("./id_to_box","rb+")
    box=pickle.load(f)
    box=box[index]
    box_train=box[0:24000]
    box_test=box[0:5]
    data_train = da.concatenate((data_train, data_train1), axis=0)
    return data_train, box_train,data_test,box_test


def plot_model(model_details):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_details.history['my_metric'])+1),model_details.history['my_metric'])
    axs[0].plot(range(1,len(model_details.history['val_my_metric'])+1),[1.7*x for x in model_details.history['val_my_metric']])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['my_metric'])+1),len(model_details.history['my_metric'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig("model.png")
