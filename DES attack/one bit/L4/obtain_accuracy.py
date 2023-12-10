# -*- coding: utf-8 -*-



import numpy as np
from os import urandom

#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#
import pickle
import tensorflow as tf  
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  
config.gpu_options.allow_growth=True  
session = tf.compat.v1.Session(config=config)  
tf.compat.v1.keras.backend.set_session(session)#
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import model_from_json



import des_for_linear as ciph
#only for 4-round linear expression of des


def get_Round():
    return 4


def make_train_data(n,num_pair):
    Round=get_Round()#
    
    
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    

    cipher_X=[]
    keys = np.frombuffer(urandom(64*n), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,Round)
    for i in range(num_pair):
        print(i)
        plain = np.frombuffer(urandom(64*n), dtype=np.uint8).reshape(-1,64)
        plain = plain & 1

        Ph=np.array(plain[:,:32])
        Pl=np.array(plain[:,32:])
    
        for sk in subkey:
            Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
        cipher=np.concatenate((Pl,Ph),axis=1)
            
        Ph=np.array(plain[:,:32])
        Pl=np.array(plain[:,32:])
    
        Ch=np.array(cipher[:,:32])
        Cl=np.array(cipher[:,32:])
            
        cipher_X.append(Ph[:,31-7]^Ph[:,31-18]^Ph[:,31-24]^Ph[:,31-29]^Pl[:,31-15]^Ch[:,31-15]^Cl[:,31-7]^Cl[:,31-18]^Cl[:,31-24]^Cl[:,31-27]^Cl[:,31-28]^Cl[:,31-29]^Cl[:,31-30]^Cl[:,31-31]^subkey[0][:,47-22]^subkey[2][:,47-22]^subkey[3][:,47-42]^subkey[3][:,47-43]^subkey[3][:,47-45]^subkey[3][:,47-46])
    cipher_X=np.array(cipher_X,dtype=np.uint8)
    cipher_X=cipher_X.T
    
    error_k=np.ones((sum(Y==0),num_pair),dtype=np.uint8)

    cipher_X[Y==0]=cipher_X[Y==0]^error_k
    
    return cipher_X,Y
   

pc_pair=512

#
single_json_file = open('./neural_network/single_L_'+str(get_Round())+'_'+str(pc_pair)+'v2_model.json','r')
single_json_model = single_json_file.read()
single_net = model_from_json(single_json_model)
single_net.load_weights('./neural_network/single_L_'+str(get_Round())+'_'+str(pc_pair)+'v2_weight.h5')

n=2**20
X,Y=make_train_data(n,pc_pair)
result=single_net.predict(X,batch_size=2**18)

result=result.flatten()
result=np.around(result) 

result=np.array(result,dtype=np.int64) 
Y=np.array(Y,dtype=np.int64) 

acc=1-sum((result-Y)*(result-Y))/len(Y)
print('round='+str(get_Round()),pc_pair,acc)
   
    
