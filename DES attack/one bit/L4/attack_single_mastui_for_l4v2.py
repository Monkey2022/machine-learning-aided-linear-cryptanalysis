# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:02:22 2023

@author: x
"""


import numpy as np
from os import urandom


import des_for_linear as ciph
#only for 4-round linear expression of des


def get_Round():
    return 4


def obtain_subkey_use_L4(p,c):
    
    pr=0.5-1.95*pow(2,-5)
   
    
    plain=p
    cipher=c
    
    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])
    

    
    sample=Ph[:,31-7]^Ph[:,31-18]^Ph[:,31-24]^Ph[:,31-29]^Pl[:,31-15]^Ch[:,31-15]^Cl[:,31-7]^Cl[:,31-18]^Cl[:,31-24]^Cl[:,31-27]^Cl[:,31-28]^Cl[:,31-29]^Cl[:,31-30]^Cl[:,31-31]
    
    sample=sample.flatten()
    sample=np.array(sample,dtype=np.uint32)
    
    T=len(sample)-sum(sample)
    if(T>0.5*len(sample)):
        if(pr>0.5):
            return 0
        else:
            return 1
    else:
        if(pr>0.5):
            return 1
        else:
            return 0

def attack_4_round_use_L4(num_pair):

    

    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,get_Round())#
    

    plain = np.frombuffer(urandom(64*num_pair), dtype=np.uint8).reshape(-1,64)
    plain = plain & 1

    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])

    for sk in subkey:
        Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
    cipher=np.concatenate((Pl,Ph),axis=1)#
    
    real_key=subkey[0][:,47-22]^subkey[2][:,47-22]^subkey[3][:,47-42]^subkey[3][:,47-43]^subkey[3][:,47-45]^subkey[3][:,47-46]
    real_key=real_key[0]
    
    
    guess_key=obtain_subkey_use_L4(plain,cipher)

    return real_key,guess_key
    

num=10**5
flag=0


Real_key=[]
Guess_key=[]


num_pair=512

NUM_pair=[num_pair]

for num_pair in NUM_pair:

    for i in range(num): 
    
        real_key,guess_key=attack_4_round_use_L4(num_pair)
        Real_key.append(real_key)
        Guess_key.append(guess_key)
        np.save('real_key_for_mastui'+str(num_pair)+'.npy',np.array(Real_key,dtype=np.uint8))
        np.save('guess_key_for_mastui'+str(num_pair)+'.npy',np.array(Guess_key,dtype=np.uint8))
        if(real_key==guess_key):
            flag=flag+1
        print(i,flag/(i+1))

print(flag/num)
    
    
    