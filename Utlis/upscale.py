from keras import backend as K
import numpy as np
import tensorflow as tf
from Dilationlayer import Dilationlayer,CNNlayer
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,GlobalMaxPooling2D,SpatialDropout2D,Conv2DTranspose,UpSampling2D
import os
from InceptionA import InceptionA
from InceptionB import InceptionB
from InceptionC import InceptionC
from InceptionD import InceptionD
from InceptionE import InceptionE
from tensorflow.keras.layers import  BatchNormalization,Activation,concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Lambda, multiply,Multiply,Dot
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# #The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# #The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,5,2"
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Cropping2D, Attention

class Upscale():
    def __init__(self):
        self.alpha = 0.3
        
        

        

    def multiply1(self,Inputlayer):
#         x= self.inputlayer
        trans = Conv2DTranspose(filters = 192, kernel_size =(1,1))(Inputlayer)
        
#         trans = lambda x: tf.transpose(x,[0,1,2,3])
#         newtrans = Lambda(trans)(x)
#         transn = Flatten()(x)

#         transposelayer = newtrans
        print("<<transpose>>")
        print(trans)
#         trans = lambda x: K.permute_dimensions(x,(0,3,1,2))
#         outputtarns = lambda x: tf.dot(x,newtrans)
#         output = Lambda(outputtarns)(x)
        output = Attention()([Inputlayer,trans])
#         output = Dot(1,2)([transposelayer,self.inputlayer,])
        print("<<mul>>")
        print(output)
        output1 = UpSampling2D((17, 17))(output)
#         output1 = Reshape((256, 256,2048))(output)
        print("<<matrixmap>>")
        print(output1)
        output2 = Cropping2D(cropping=((120,120), (120, 120)),
                         )(output1)
#         print("<<Cropped map>>")
#         print(output2)
#         output3 = Reshape((18, 18,2048))(output2)
#         print("<<final map>>")
#         print(output3)
#         with tf.compat.v1.variable_scope('al',reuse=tf.AUTO_REUSE):
#         init=tf.global_variables_initializer()
#         with tf.Session() as sess:
#             sess.run(init)
    
#         alpha = tf.Variable(initial_value=0.0, trainable=True)
        
#         Newlayer = Multiply()([lambda x: x*self.alpha,(output3)])
#         output3 = Lambda(lambda x: x * self.alpha)(output2)
#         print("<<final layer>>")
        print(output2)
        return output2
        
#         if K.image_dim_ordering() == 'th':
#             features = K.batch_flatten(x)
#         else:
#             features = K.batch_flatten(K.permute_dimensions(x, (3, 0,1, 2)))
#         return K.dot(features, K.transpose(features))
        