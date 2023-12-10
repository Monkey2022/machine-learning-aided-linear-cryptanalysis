# -*- coding: utf-8 -*-

import numpy as np
from os import urandom


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#
import pickle
import tensorflow as tf  
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  
config.gpu_options.allow_growth=True  
session = tf.compat.v1.Session(config=config)  
tf.compat.v1.keras.backend.set_session(session)#
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import model_from_json
from math import log2
from time import time



import des_for_linear as ciph
#only for 6-round linear expression of des


def num2bitarray6(x):
    r=bin(x)[2:]
    r=r.zfill(6)
    r=list(r)
    r=np.array(r,dtype=np.uint8)
    return r

def array2num(x):
    r=np.array(x,dtype=np.uint32)
    l=len(r)
    res=0
    for i in range(l):
        res=res+(r[i]<<(l-1-i))
    return res

def bit2num(x):
    return (x[:,0]<<5)+(x[:,1]<<4)+(x[:,2]<<3)+(x[:,3]<<2)+(x[:,4]<<1)+(x[:,5]<<0)


def get_Round():
    return 6


def S_box():
    s0=np.array([[1,1,1,0],[0,0,0,0],[0,1,0,0],[1,1,1,1],[1,1,0,1],[0,1,1,1],[0,0,0,1],[0,1,0,0],[0,0,1,0],[1,1,1,0],[1,1,1,1],[0,0,1,0],[1,0,1,1],[1,1,0,1],[1,0,0,0],[0,0,0,1],[0,0,1,1],[1,0,1,0],[1,0,1,0],[0,1,1,0],[0,1,1,0],[1,1,0,0],[1,1,0,0],[1,0,1,1],[0,1,0,1],[1,0,0,1],[1,0,0,1],[0,1,0,1],[0,0,0,0],[0,0,1,1],[0,1,1,1],[1,0,0,0],[0,1,0,0],[1,1,1,1],[0,0,0,1],[1,1,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[0,0,1,0],[1,1,0,1],[0,1,0,0],[0,1,1,0],[1,0,0,1],[0,0,1,0],[0,0,0,1],[1,0,1,1],[0,1,1,1],[1,1,1,1],[0,1,0,1],[1,1,0,0],[1,0,1,1],[1,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,0],[0,0,1,1],[1,0,1,0],[1,0,1,0],[0,0,0,0],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,1,0,1]],dtype=np.uint8)
    s1=np.array([[1,1,1,1],[0,0,1,1],[0,0,0,1],[1,1,0,1],[1,0,0,0],[0,1,0,0],[1,1,1,0],[0,1,1,1],[0,1,1,0],[1,1,1,1],[1,0,1,1],[0,0,1,0],[0,0,1,1],[1,0,0,0],[0,1,0,0],[1,1,1,0],[1,0,0,1],[1,1,0,0],[0,1,1,1],[0,0,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,1],[1,0,1,0],[1,1,0,0],[0,1,1,0],[0,0,0,0],[1,0,0,1],[0,1,0,1],[1,0,1,1],[1,0,1,0],[0,1,0,1],[0,0,0,0],[1,1,0,1],[1,1,1,0],[1,0,0,0],[0,1,1,1],[1,0,1,0],[1,0,1,1],[0,0,0,1],[1,0,1,0],[0,0,1,1],[0,1,0,0],[1,1,1,1],[1,1,0,1],[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,1],[1,0,1,1],[1,0,0,0],[0,1,1,0],[1,1,0,0],[0,1,1,1],[0,1,1,0],[1,1,0,0],[1,0,0,1],[0,0,0,0],[0,0,1,1],[0,1,0,1],[0,0,1,0],[1,1,1,0],[1,1,1,1],[1,0,0,1]],dtype=np.uint8)
    s2=np.array([[1,0,1,0],[1,1,0,1],[0,0,0,0],[0,1,1,1],[1,0,0,1],[0,0,0,0],[1,1,1,0],[1,0,0,1],[0,1,1,0],[0,0,1,1],[0,0,1,1],[0,1,0,0],[1,1,1,1],[0,1,1,0],[0,1,0,1],[1,0,1,0],[0,0,0,1],[0,0,1,0],[1,1,0,1],[1,0,0,0],[1,1,0,0],[0,1,0,1],[0,1,1,1],[1,1,1,0],[1,0,1,1],[1,1,0,0],[0,1,0,0],[1,0,1,1],[0,0,1,0],[1,1,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,1],[0,0,0,1],[0,1,1,0],[1,0,1,0],[0,1,0,0],[1,1,0,1],[1,0,0,1],[0,0,0,0],[1,0,0,0],[0,1,1,0],[1,1,1,1],[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,0],[0,1,1,1],[1,0,1,1],[0,1,0,0],[0,0,0,1],[1,1,1,1],[0,0,1,0],[1,1,1,0],[1,1,0,0],[0,0,1,1],[0,1,0,1],[1,0,1,1],[1,0,1,0],[0,1,0,1],[1,1,1,0],[0,0,1,0],[0,1,1,1],[1,1,0,0]],dtype=np.uint8)
    s3=np.array([[0,1,1,1],[1,1,0,1],[1,1,0,1],[1,0,0,0],[1,1,1,0],[1,0,1,1],[0,0,1,1],[0,1,0,1],[0,0,0,0],[0,1,1,0],[0,1,1,0],[1,1,1,1],[1,0,0,1],[0,0,0,0],[1,0,1,0],[0,0,1,1],[0,0,0,1],[0,1,0,0],[0,0,1,0],[0,1,1,1],[1,0,0,0],[0,0,1,0],[0,1,0,1],[1,1,0,0],[1,0,1,1],[0,0,0,1],[1,1,0,0],[1,0,1,0],[0,1,0,0],[1,1,1,0],[1,1,1,1],[1,0,0,1],[1,0,1,0],[0,0,1,1],[0,1,1,0],[1,1,1,1],[1,0,0,1],[0,0,0,0],[0,0,0,0],[0,1,1,0],[1,1,0,0],[1,0,1,0],[1,0,1,1],[0,0,0,1],[0,1,1,1],[1,1,0,1],[1,1,0,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[0,0,0,1],[0,1,0,0],[0,0,1,1],[0,1,0,1],[1,1,1,0],[1,0,1,1],[0,1,0,1],[1,1,0,0],[0,0,1,0],[0,1,1,1],[1,0,0,0],[0,0,1,0],[0,1,0,0],[1,1,1,0]],dtype=np.uint8)
    s4=np.array([[0,0,1,0],[1,1,1,0],[1,1,0,0],[1,0,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0],[0,1,1,1],[0,1,0,0],[1,0,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1],[0,1,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,1],[0,1,0,1],[0,0,0,0],[0,0,1,1],[1,1,1,1],[1,1,1,1],[1,0,1,0],[1,1,0,1],[0,0,1,1],[0,0,0,0],[1,0,0,1],[1,1,1,0],[1,0,0,0],[1,0,0,1],[0,1,1,0],[0,1,0,0],[1,0,1,1],[0,0,1,0],[1,0,0,0],[0,0,0,1],[1,1,0,0],[1,0,1,1],[0,1,1,1],[1,0,1,0],[0,0,0,1],[1,1,0,1],[1,1,1,0],[0,1,1,1],[0,0,1,0],[1,0,0,0],[1,1,0,1],[1,1,1,1],[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,1,0,0],[0,0,0,0],[0,1,0,1],[1,0,0,1],[0,1,1,0],[1,0,1,0],[0,0,1,1],[0,1,0,0],[0,0,0,0],[0,1,0,1],[1,1,1,0],[0,0,1,1]],dtype=np.uint8)
    s5=np.array([[1,1,0,0],[1,0,1,0],[0,0,0,1],[1,1,1,1],[1,0,1,0],[0,1,0,0],[1,1,1,1],[0,0,1,0],[1,0,0,1],[0,1,1,1],[0,0,1,0],[1,1,0,0],[0,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,1,0],[1,1,0,1],[0,0,0,1],[0,0,1,1],[1,1,0,1],[0,1,0,0],[1,1,1,0],[1,1,1,0],[0,0,0,0],[0,1,1,1],[1,0,1,1],[0,1,0,1],[0,0,1,1],[1,0,1,1],[1,0,0,0],[1,0,0,1],[0,1,0,0],[1,1,1,0],[0,0,1,1],[1,1,1,1],[0,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,1],[1,1,0,0],[1,1,1,1],[0,0,1,1],[1,0,1,0],[0,1,1,1],[1,0,1,1],[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,1],[1,0,1,0],[0,1,1,1],[0,0,0,1],[0,1,1,0],[1,1,0,1],[0,0,0,0],[1,0,1,1],[1,0,0,0],[0,1,1,0],[1,1,0,1]],dtype=np.uint8)
    s6=np.array([[0,1,0,0],[1,1,0,1],[1,0,1,1],[0,0,0,0],[0,0,1,0],[1,0,1,1],[1,1,1,0],[0,1,1,1],[1,1,1,1],[0,1,0,0],[0,0,0,0],[1,0,0,1],[1,0,0,0],[0,0,0,1],[1,1,0,1],[1,0,1,0],[0,0,1,1],[1,1,1,0],[1,1,0,0],[0,0,1,1],[1,0,0,1],[0,1,0,1],[0,1,1,1],[1,1,0,0],[0,1,0,1],[0,0,1,0],[1,0,1,0],[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,1],[0,1,1,0],[0,0,0,1],[0,1,1,0],[0,1,0,0],[1,0,1,1],[1,0,1,1],[1,1,0,1],[1,1,0,1],[1,0,0,0],[1,1,0,0],[0,0,0,1],[0,0,1,1],[0,1,0,0],[0,1,1,1],[1,0,1,0],[1,1,1,0],[0,1,1,1],[1,0,1,0],[1,0,0,1],[1,1,1,1],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,1,1,1],[0,0,0,0],[1,1,1,0],[0,1,0,1],[0,0,1,0],[1,0,0,1],[0,0,1,1],[0,0,1,0],[1,1,0,0]],dtype=np.uint8)
    s7=np.array([[1,1,0,1],[0,0,0,1],[0,0,1,0],[1,1,1,1],[1,0,0,0],[1,1,0,1],[0,1,0,0],[1,0,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,1],[0,0,1,1],[1,0,1,1],[0,1,1,1],[0,0,0,1],[0,1,0,0],[1,0,1,0],[1,1,0,0],[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,1,1,0],[1,1,1,0],[1,0,1,1],[0,1,0,1],[0,0,0,0],[0,0,0,0],[1,1,1,0],[1,1,0,0],[1,0,0,1],[0,1,1,1],[0,0,1,0],[0,1,1,1],[0,0,1,0],[1,0,1,1],[0,0,0,1],[0,1,0,0],[1,1,1,0],[0,0,0,1],[0,1,1,1],[1,0,0,1],[0,1,0,0],[1,1,0,0],[1,0,1,0],[1,1,1,0],[1,0,0,0],[0,0,1,0],[1,1,0,1],[0,0,0,0],[1,1,1,1],[0,1,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,0,1],[0,0,0,0],[1,1,1,1],[0,0,1,1],[0,0,1,1],[0,1,0,1],[0,1,0,1],[0,1,1,0],[1,0,0,0],[1,0,1,1]],dtype=np.uint8)
    
    return [s0,s1,s2,s3,s4,s5,s6,s7]

def obtain_subkey_use_L6(p,c,sample_pair,pc_pair):
    
    s_box=S_box()
    
    #load neural distinguishers
    mul_json_file = open('multiple_L_'+str(get_Round())+'_'+str(pc_pair)+'_model.json','r')
    mul_json_model = mul_json_file.read()
    mul_net = model_from_json(mul_json_model)
    mul_net.load_weights('multiple_L_'+str(get_Round())+'_'+str(pc_pair)+'_weight.h5')
    
    plain=p
    cipher=c
    
    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])
    
    score_k0=[]
    score_k1=[]
    
    
    #guess the subky in the first round
    for k1 in range(2**6):
        Guess_k1=num2bitarray6(k1)
        Pl_extend=Pl[:,[15,16, 17,18,19,20]]#
        
        Guess_k1=np.tile(Guess_k1,reps=(len(Pl_extend),1))
        
        
        Pl_extend=Pl_extend^Guess_k1
        
        buffer_k1=bit2num(Pl_extend)
        buffer_k1=s_box[4][buffer_k1]
        
        ahead_part=Ph[:,31-7]^Ph[:,31-18]^Ph[:,31-24]^buffer_k1[:,0]^buffer_k1[:,1]^buffer_k1[:,2]
        
        #guess the last subkey
        for k5 in range(2**6):
            Guess_k5=num2bitarray6(k5)
            Cl_extend=Cl[:,[31,0,1,2,3,4]]#
            
            Guess_k5=np.tile(Guess_k5,reps=(len(Cl_extend),1))
            
            Cl_extend=Cl_extend^Guess_k5
            
            buffer_k5=bit2num(Cl_extend)
            buffer_k5=s_box[0][buffer_k5]
            
            behind_part=Cl[:,31-7]^Cl[:,31-18]^Cl[:,31-24]^Cl[:,31-29]^Ch[:,31-15]^buffer_k5[:,1]
            
            sample=behind_part^ahead_part
            
            sample0=np.array(sample)
            sample0=sample0.reshape(-1,pc_pair)
            result0=mul_net.predict(np.array(sample0),batch_size=2**18)
            result0=result0.flatten()
            result0=result0/(1-result0)
            result0 = np.log2(result0)
            score_k0.append(sum(result0))
            
            sample1=np.array(sample^1)
            sample1=sample1.reshape(-1,pc_pair)
            result1=mul_net.predict(np.array(sample1),batch_size=2**18)
            result1=result1.flatten()
            result1=result1/(1-result1)
            result1 = np.log2(result1) 
            score_k1.append(sum(result1))

    score_k=score_k0+score_k1

    guess_key=np.where(np.array(score_k,dtype=np.float32)==max(np.array(score_k,dtype=np.float32)))[0][0]
    
    guess_key=bin(guess_key)[2:]
    guess_key=guess_key.zfill(13)
    guess_key=list(guess_key)
    guess_key=np.array(guess_key,dtype=np.uint8)
    
    return guess_key


def attack_8_round_use_L6():
    Sample_pair=5000#number of samples 
    Pc_pair=80#ength of sample
    
    ##generate a master key
    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,1+get_Round()+1)#1+6+1-round attack
    
    ##generate Sample_pair*Pc_pair plaintexts
    plain = np.frombuffer(urandom(64*Sample_pair*Pc_pair), dtype=np.uint8).reshape(-1,64)
    plain = plain & 1

    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    #encrypt these plaintext using the master key
    for sk in subkey:
        Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
    cipher=np.concatenate((Pl,Ph),axis=1)#
    
    real_key=np.zeros(13,dtype=np.uint8)
    
    #recover 6+6+1 bits
    #1 recover k3[22]+k4[44]+k5[22]+k7[22]
    #2 K1[18,19,20,21,22,23]
    #3 K8[41,42,43,44,45,46]
    real_key[0]=subkey[2][0,47-22]^subkey[3][0,47-44]^subkey[4][0,47-22]^subkey[6][0,47-22]
    real_key[1]=subkey[0][0,24]
    real_key[2]=subkey[0][0,25]
    real_key[3]=subkey[0][0,26]
    real_key[4]=subkey[0][0,27]
    real_key[5]=subkey[0][0,28]
    real_key[6]=subkey[0][0,29]
    
    real_key[7]=subkey[7][0,0]
    real_key[8]=subkey[7][0,1]
    real_key[9]=subkey[7][0,2]
    real_key[10]=subkey[7][0,3]
    real_key[11]=subkey[7][0,4]
    real_key[12]=subkey[7][0,5]
    print(Sample_pair,Pc_pair)
    guess_key=obtain_subkey_use_L6(plain,cipher,Sample_pair,Pc_pair)
    
    
    print(real_key,end='----')
    print(guess_key)
    
    return real_key,guess_key
     
num=1000
flag=0

for i in range(num):  
    t1=time()
    real_key,guess_key=attack_8_round_use_L6()

    r=array2num(real_key)
    g=array2num(guess_key)
    if(r==g):
        flag=flag+1
    t2=time()
    print(i,flag/(i+1),t2-t1)

print(flag/num)

    
    