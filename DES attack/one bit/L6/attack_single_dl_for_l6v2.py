# -*- coding: utf-8 -*-
import numpy as np
from os import urandom


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#

import tensorflow as tf  
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  
config.gpu_options.allow_growth=True  
session = tf.compat.v1.Session(config=config)  
tf.compat.v1.keras.backend.set_session(session)#

from tensorflow.keras.models import model_from_json




import des_for_linear as ciph
#only for 6-round linear expression of des


def get_Round():
    return 6


def obtain_subkey_use_L6(p,c,sample_pair,pc_pair):

    single_json_file = open('./neural_network/single_L_'+str(get_Round())+'_'+str(pc_pair)+'v2_model.json','r')
    single_json_model = single_json_file.read()
    single_net = model_from_json(single_json_model)
    single_net.load_weights('./neural_network/single_L_'+str(get_Round())+'_'+str(pc_pair)+'v2_weight.h5')
    
    plain=p
    cipher=c
    
    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])
    

    #Guess_expression_key=0
    
    sample=Pl[:,31-7]^Pl[:,31-18]^Pl[:,31-24]^Ch[:,31-7]^Ch[:,31-18]^Ch[:,31-24]^Ch[:,31-29]^Cl[:,31-15]
    
    sample=sample.reshape(-1,pc_pair)
    
    result=single_net.predict(sample,batch_size=2**18)
    
    result=result.flatten()
    result=result/(1-result)
    result = np.log2(result)
    result=np.mean(result)
    
    if(result<0):
        return 1
    else:
        return 0
    
 


def attack_6_round_use_L6(pc_pair,sample_pair):

    
    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,get_Round())#
    

    plain = np.frombuffer(urandom(64*sample_pair*pc_pair), dtype=np.uint8).reshape(-1,64)
    plain = plain & 1

    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])

    for sk in subkey:
        Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
    cipher=np.concatenate((Pl,Ph),axis=1)#
    
    real_key=subkey[1][:,47-22]^subkey[2][:,47-44]^subkey[3][:,47-22]^subkey[5][:,47-22]
    real_key=real_key[0]
    
    
    guess_key=obtain_subkey_use_L6(plain,cipher,sample_pair,pc_pair)
    

    return real_key,guess_key
    
pr=0.5-1.95*pow(2,-9)
num_pair=pow(abs(pr-0.5),-2)


num=10**4
Pc_pair=[16,34,36,38,40,42,44,46,48,50,52,60,80,90,100,110,120,130,140,200]
Sample_pair=[1]

for pc_pair in Pc_pair:
    for sample_pair in Sample_pair:
        
        flag=0
        
        Real_key=[]
        Guess_key=[]

        for i in range(num):   
            real_key,guess_key=attack_6_round_use_L6(pc_pair,int(num_pair/pc_pair))
            Real_key.append(real_key)
            Guess_key.append(guess_key)
            if(real_key==guess_key):
                flag=flag+1
            print(i,flag/(i+1))
        print(flag/num)
        np.save('real_key_for_dl_dim='+str(pc_pair)+'_'+str(int(num_pair/pc_pair))+'.npy',np.array(Real_key,dtype=np.uint8))
        np.save('guess_key_for_dl_dim='+str(pc_pair)+'_'+str(int(num_pair/pc_pair))+'.npy',np.array(Guess_key,dtype=np.uint8))
    
    
    