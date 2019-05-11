import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import random
import tensorflow as tf
import csv
import pandas as pd
from PIL import Image



plt.switch_backend('agg')
def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image





df1 = pd.read_csv("test.csv")
sample_df = df1
fields = ['image_name', 'x1', 'x2', 'y1', 'y2']
df = pd.DataFrame(columns = fields)
mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
j=0

def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i]*224)
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.where(condition,small_res,large_res)
    return tf.reduce_mean(loss)

def my_metric(labels,predictions):
    threshhold=0.75
    x=predictions[:,0]*224
    x=tf.maximum(tf.minimum(x,224.0),0.0)
    y=predictions[:,1]*224
    y=tf.maximum(tf.minimum(y,224.0),0.0)
    width=predictions[:,2]*224
    width=tf.maximum(tf.minimum(width,224.0),0.0)
    height=predictions[:,3]*224
    height=tf.maximum(tf.minimum(height,224.0),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.multiply(width,height)
    a2=tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.reduce_mean(sum)

id_to_data={}
model = load_model('./model.h5', custom_objects={'smooth_l1_loss': smooth_l1_loss, 'my_metric': my_metric})
index=[i for i in range(0, 12195)]
for i in index: 
    path=sample_df.iloc[i]['image_name']
    print(i)
    image=Image.open("./data/images/"+path) 
    image=image.resize((224,224))
    image=np.array(image,dtype=np.uint32)
    image=image/255
    image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
    id_to_data[int(i)]=image
id_to_data1=np.array(list(id_to_data.values()))
result=model.predict(id_to_data1[index,:,:,:])

mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
j=0


fields = ['image_name', 'x1', 'x2', 'y1', 'y2']
df = pd.DataFrame(columns = fields)
for i in index:
    print("Predicting "+str(i)+"th image.")
    image=id_to_data1[i]
    prediction=result[j]
    j+=1
    for channel in range(3):
        image[:,:,channel]=image[:,:,channel]*std[channel]+mean[channel]
    image=image*255
    image=image.astype(np.uint32)
    plt.imshow(image)
    df.loc[len(df)+1] = [sample_df.iloc[i]['image_name'], prediction[0]*640, (prediction[0] + prediction[2])*640, prediction[1]*480, (prediction[1] + prediction[3])*480] #change
    plt.show()
    plt.cla()
df.to_csv("results.csv", index=False)


