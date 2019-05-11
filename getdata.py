from PIL import Image
import numpy as np
import pickle
import pandas as pd

##read-----------------------------------------------
df1 = pd.read_csv("training_set.csv")
sample_df = df1

##data preprocessing----------------------------------
def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

id_to_data={}
id_to_size={}


for i in range(0, 2400): #13999
            path=sample_df.iloc[i]['image_name']
            print(i)
            image=Image.open("./dataset/images/"+path)
            id_to_size[i]=np.array(image).shape[0:2]


id_to_size=np.array(list(id_to_size.values()))
f=open("./id_to_size","wb+")
pickle.dump(id_to_size,f,protocol=4)

id_to_box={}
id_to_size = pickle.load( open( "./id_to_size", "rb" ) )

width=[]
height=[]

for i in range(0,2400): #13999

    box = np.array([sample_df.iloc[i][j] for j in range(1,5)], dtype=np.int32)#np.float32
    width=box[1]-box[0]
    height=box[3]-box[2]
    box[0]=box[0]/id_to_size[int(i)-1][1]*224
    box[1]=box[2]/id_to_size[int(i)-1][0]*224
    box[2]=width/id_to_size[int(i)-1][1]*224
    box[3]=height/id_to_size[int(i)-1][0]*224
    print(i)
    id_to_box[i]=box
id_to_box=np.array(list(id_to_box.values()))
f=open("./id_to_box","wb+") 
pickle.dump(id_to_box,f,protocol=4)
