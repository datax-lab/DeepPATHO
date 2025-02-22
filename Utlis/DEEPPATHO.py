import numpy as np
import tensorflow as tf
from Dilationlayer import Dilationlayer,CNNlayer
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,GlobalMaxPooling2D,SpatialDropout2D
import os
from InceptionA import InceptionA
from InceptionB import InceptionB
from InceptionC import InceptionC
from InceptionD import InceptionD
from InceptionE import InceptionE
from tensorflow.keras.layers import  BatchNormalization,Activation,concatenate, Add
from tensorflow.keras.regularizers import l2
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# #The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# #The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,5,2"
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Cropping2D,Reshape
from keras import backend as K 
from upscale import Upscale
from downscale import Downscale
class DeepPATHO_core():

#     def __init__(self):
#         self.name = name

        
        
    def bulid(self,input_img1,input_img2):
        modelt1  = Dilationlayer("20xTrack"+"baselayer").bulid(input_img1)
        modelt2  = Dilationlayer("5xTrack"+"baselayer").bulid(input_img2)
        modelt1 = Dropout(0.2)(modelt1)
        modelt2 = Dropout(0.7)(modelt2)
        # model = Dilationlayer(self.name + "baselayer1").bulid(input_img)
        # model = Dropout(0.2)(model)
        # model = Dilationlayer(self.name + "baselayer1").bulid(model)
        # with tf.device('/device:GPU:1'):
        # model = Dropout(0.25)(model)
        modelt1  = InceptionA("20xTrack"+"mixedA1").bulid(modelt1)
        modelt2  = InceptionA("5xTrack"+"mixedA1").bulid(modelt2)
        modelt1 = Dropout(0.2)(modelt1)
        modelt2 = Dropout(0.7)(modelt2)
        # # model = Dropout(0.2)(model)
        # # # model = Dropout(0.25)(model)
        # # model  = InceptionA(self.name+"mixedA2").bulid(model)
        # # # model  = InceptionA(self.name+"mixedA3").bulid(model)
        modelt1  = InceptionB("20xTrack"+"mixedB1").bulid(modelt1)
        modelt2  = InceptionB("5xTrack"+"mixedB1").bulid(modelt2)
        modelt1 = Dropout(0.2)(modelt1)
        modelt2 = Dropout(0.7)(modelt2)
        # # model = Dropout(0.2)(model)
        # # # model = Dropout(0.25)(model)
        # # # with tf.device('/device:GPU:2'):
#         model  = InceptionC(self.name+"mixedC1",128).bulid(model)
#         model  = InceptionC(self.name+"mixedC2",160).bulid(model)
        # model = Dropout(0.25)(model)
#         model  = InceptionC(self.name+"mixedC3",160).bulid(model)
#         model  = InceptionC(self.name+"mixedC4",192).bulid(model)
        # # with tf.device('/device:GPU:3'):
        modelt1  = InceptionD("20xTrack"+"mixedD1").bulid(modelt1)
        modelt2  = InceptionD("5xTrack"+"mixedD1").bulid(modelt2)
        # # # model = Dropout(0.25)(model)
#         modelt1 = Dropout(0.3)(modelt1)
#         modelt2 = Dropout(0.7)(modelt2)
        modelt1  = InceptionE("20xTrack"+"mixedE1").bulid(modelt1)
        modelt1 = Dropout(0.3)(modelt1)
        modelt2  = InceptionE("5xTrack"+"mixedE1").bulid(modelt2)
        modelt2 = Dropout(0.7)(modelt2)
        print("<<orginal>>")
        print(modelt1)
        print("<<image_order>>")
        print(K.image_dim_ordering())
#         print("<<transpose>>")
        f = Upscale().multiply1(modelt2)
        g = Downscale().multiply1(modelt1)
        modelt1a = Add()([0.2*f,0.8*modelt1])
        modelt2a = Add()([0.3*g,0.7*modelt2])
        print("<<finalorder>>")
        print(modelt1a)
        print(modelt2)
        modelt1a = Dropout(0.3)(modelt1a)
        modelt2a = Dropout(0.7)(modelt2a)
        model3_l = Cropping2D(cropping=((5,6), (5, 6)),
                         )(modelt2a)
        model4_l = Reshape((4, 4,192))(modelt1a)
        print(model3_l)
        print(model4_l)
        CATNet_Track1flat = GlobalAveragePooling2D()(modelt1a)
        CATNet_Track2flat =  GlobalAveragePooling2D()(modelt2a)
#         print("checksimilarity")
        CATNET_Track1similarity  = GlobalAveragePooling2D()(model4_l)
        CATNET_Track2similarity  = GlobalAveragePooling2D()(model3_l)
        print(CATNet_Track1flat,CATNet_Track2flat)
        print("new")
        print(CATNET_Track1similarity,CATNET_Track2similarity)
#         model = Dropout(0.4)(model)
        combined = concatenate([CATNet_Track1flat, CATNet_Track2flat])
        # model  = InceptionE(self.name+"mixedE2").bulid(model)
#         modelt1a = Dropout(0.4)(modelt1a)
        # with tf.device('/device:GPU:'):
        # model = Dropout(0.4)(model)
        model = Model([input_img1,input_img2],[combined,CATNET_Track1similarity,CATNET_Track2similarity])
        
        return model,model3_l,model4_l
class DeepPATHO():
    
    def __init__(self, name):
        
        self.name = name

        

    def bulid(InputA,InputB):
        # with tf.device('/device:GPU:0'):
        
        CATNet_Track1,model3_l,model4_l = DeepPATHO_core().bulid(InputA,InputB)
        CATNet_Track1output = CATNet_Track1.output[0]
        CATNET_Track1similarity = CATNet_Track1.output[1]
        CATNET_Track2similarity = CATNet_Track1.output[2]
        print(CATNET_Track1similarity)
# #         CATNet_Track2 = CATNet("5xTrack").bulid(InputB)

# #         CATNet_Track2output = CATNet_Track2.output
      

#         # CATNet_Track2 = Dropout(0.4)(CATNet_Track2)
#         CATNet_Track1flat = GlobalAveragePooling2D()(CATNet_Track1output)
#         # CATNet_Track1flat = Flatten()(CATNet_Track1output)
#         # CATNet_Track2flat = Flatten()(CATNet_Track2output)

#         CATNet_Track2flat =  GlobalAveragePooling2D()(CATNet_Track2output)
#         # CATNet_Track2flat = Flatten()(CATNet_Track2flat)
#         combined = concatenate([CATNet_Track1flat, CATNet_Track2flat])
        # combinedflat = GlobalAveragePooling1D()(combined)
        # combined = Flatten()(combined)
        z = Dense(4096, activation='tanh',kernel_regularizer = l2(0.4))(CATNet_Track1output)
        z = Dense(1024, activation='tanh',kernel_regularizer = l2(0.4))(z)
        z = Dense(152, activation='tanh',kernel_regularizer = l2(0.4))(z)
        # z = Dense(512, activation='tanh')(z)
        z = Dropout(0.4)(z)
        predictions = Dense(1, activation='sigmoid')(z)
        model = Model(inputs=CATNet_Track1.input, outputs=predictions)
        return model,CATNET_Track1similarity,CATNET_Track2similarity
