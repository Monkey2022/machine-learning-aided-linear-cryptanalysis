# -*- coding: utf-8 -*-
import numpy as np
from os import urandom


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf  
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  
config.gpu_options.allow_growth=True  
session = tf.compat.v1.Session(config=config)  
tf.compat.v1.keras.backend.set_session(session)#

from tensorflow.keras.models import model_from_json

import des_for_linear as ciph



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
    return 4


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

def obtain_subkey_use_L4(p,c,sample_pair,pc_pair):
    
    s_box=S_box()
    
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
   
    
    #猜测
    for k1 in range(1):
        
        ahead_part=Cl[:,31-7]^Cl[:,31-18]^Cl[:,31-24]^Cl[:,31-29]^Cl[:,31-27]^Cl[:,31-28]^Cl[:,31-30]^Cl[:,31-31]^Ch[:,31-15]
        
        for k5 in range(2**6):
            Guess_k5=num2bitarray6(k5)
            Pl_extend=Pl[:,[31,0,1,2,3,4]]#
            
            for j in range(len(Pl_extend)):
                Pl_extend[j]=Pl_extend[j]^Guess_k5
            buffer_k5=bit2num(Pl_extend)
            buffer_k5=s_box[0][buffer_k5]
            
            behind_part=Pl[:,31-7]^Pl[:,31-18]^Pl[:,31-24]^Pl[:,31-29]^Ph[:,31-15]^buffer_k5[:,1]
            
            sample0=behind_part^ahead_part
            
            sample0=sample0.reshape(-1,pc_pair)
            
            result0=mul_net.predict(np.array(sample0),batch_size=2**18)
            result0=result0.flatten()
            result0=result0/(1-result0)
            result0 = np.log2(result0)
            
            score_k0.append(sum(result0))
            
            sample1=behind_part^ahead_part^1
            
            sample1=sample1.reshape(-1,pc_pair)
            
            result1=mul_net.predict(np.array(sample1),batch_size=2**18)
            result1=result1.flatten()
            result1=result1/(1-result1)
            result1 = np.log2(result1)
            
            score_k1.append(sum(result1))


    score_k=score_k0+score_k1
    guess_key=np.where(np.array(score_k,dtype=np.float32)==max(np.array(score_k,dtype=np.float32)))[0][0]
    
    guess_key=bin(guess_key)[2:]
    guess_key=guess_key.zfill(7)
    guess_key=list(guess_key)
    guess_key=np.array(guess_key,dtype=np.uint8)
    
    return guess_key,score_k


def attack_5_round_use_L4(Pc_pair,Sample_pair):
    #
    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,get_Round()+1)#
    plain = np.frombuffer(urandom(64*Sample_pair*Pc_pair), dtype=np.uint8).reshape(-1,64)
    plain = plain & 1

    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])

    for sk in subkey:
        Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
    cipher=np.concatenate((Pl,Ph),axis=1)#
    
    real_key=np.zeros(7,dtype=np.uint8)
    
    real_key[0]=subkey[1][0,47-22]^subkey[3][0,47-22]^subkey[4][0,47-42]^subkey[4][0,47-43]^subkey[4][0,47-45]^subkey[4][0,47-46]

    real_key[1]=subkey[0][0,0]
    real_key[2]=subkey[0][0,1]
    real_key[3]=subkey[0][0,2]
    real_key[4]=subkey[0][0,3]
    real_key[5]=subkey[0][0,4]
    real_key[6]=subkey[0][0,5]
    
    guess_key,key_score=obtain_subkey_use_L4(plain,cipher,Sample_pair,Pc_pair)
    

    
    return real_key,guess_key,key_score
    

num_plaintext=512

pc_pair=22#length of sample
sample_pair=int(num_plaintext/pc_pair)#number of sample(number of plaintext = pc_pair*sample_pair)

num=10**3
flag=0

Real_key=[]
Guess_key=[]
Key_score=[]

for i in range(num):   
    real_key,guess_key,key_score=attack_5_round_use_L4(pc_pair,sample_pair)

    r=array2num(real_key)
    g=array2num(guess_key)
    if(r==g):
        flag=flag+1
    print(i,pc_pair,sample_pair,flag/(i+1))
    
    Real_key.append(real_key)
    Guess_key.append(guess_key) 
    Key_score.append(key_score)
    np.save(str(get_Round())+'_round_distinguisher_'+str(get_Round()+1)+'_round_attack_dim='+str(pc_pair)+'_num_plaintext='+str(sample_pair*pc_pair)+'_realkey.npy',np.array(Real_key,dtype=np.uint8))
    np.save(str(get_Round())+'_round_distinguisher_'+str(get_Round()+1)+'_round_attack_dim='+str(pc_pair)+'_num_plaintext='+str(sample_pair*pc_pair)+'_guesskey.npy',np.array(Guess_key,dtype=np.uint8))
    np.save(str(get_Round())+'_round_distinguisher_'+str(get_Round()+1)+'_round_attack_dim='+str(pc_pair)+'_num_plaintext='+str(sample_pair*pc_pair)+'_keyscore.npy',np.array(Key_score))

print(flag/num)
 