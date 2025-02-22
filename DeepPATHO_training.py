import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, help="GPU number", required=True)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

from os import walk
import pickle
import random
import numpy as np
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split,KFold

from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

from DataGenetors import ImgDataParameters,DataGenerator
from Utlis.DEEPPATHO import DeepPATHO

DATA_PATH = "/home/USERNAME/PATH/TO/DATAFOLDER/"


input_imgen = ImageDataGenerator(rescale = 1./255 )
test_imgen = ImageDataGenerator(rescale = 1./255)



def generate_generator_multiple(generator, dir1, dir2):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (256,256),
                                          class_mode="binary",
                                          
                                          
                                          color_mode="rgb",
                                          batch_size =16,
                                          shuffle=True, 
                                          seed=42)
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (256,256),
                                          class_mode="binary",
                                                                                 
                                          color_mode="rgb",
                                          batch_size =16,
                                          shuffle=True, 
                                          seed=42)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label
            
            
train_generator = generate_generator_multiple(generator=input_imgen,
                                              dir1=r"/home/USERNAME/PATH/TO/20x/IMAGES/",
                                              dir2=r"/home/USERNAME/PATH/TO/5x/IMAGES/")
     

validation_generator = generate_generator_multiple(generator=input_imgen,
                                                   dir1=r"/home/USERNAME/PATH/TO/20x/IMAGES/",
                                                   dir2=r"/home/USERNAME/PATH/TO/5x/IMAGES/")


test_generator = generate_generator_multiple(generator=test_imgen,
                                             dir1= r"/home/USERNAME/PATH/TO/20x/IMAGES/",
                                             dir2= r"/home/USERNAME/PATH/TO/5x/IMAGES/")
          

batch_size = 16
Lr = [0.001]
Beta_1 = [0.85]
Paramaters_list = []
epochs =20

for lr in  Lr:
    for beta1 in Beta_1:
        Adam1 = optimizers.Nadam(learning_rate=lr, beta_1 = beta1)        
        InputA = Input(shape=(256,256,3))
        InputB = Input(shape=(256,256,3))
        model, track1_similarity, track2_similarity = CATNet2.bulid(InputA, InputB)
        
        model.compile(optimizer = Adam1, loss = 'binary_crossentropy',metrics=['accuracy'])
        history = model.fit(traingenerator,
                        steps_per_epoch=int(96627/(16)),
                        epochs = epochs,
                        validation_data = validationgenerator,
                        validation_steps = int(22298/(16)),                    
                        use_multiprocessing=False,
                        shuffle=True)
        
        ac = model.evaluate(testgenerator, steps=int(29731/(16)))
        model.save_weights('newweightsnn1probnew.tf')
        model.save_weights('newweightsnn12probnew.h5')
        model.save("newmodelnn13probnew.h5")
        
        with open('./history/historyn6.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)