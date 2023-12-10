# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:02:22 2023

@author: x
"""


import numpy as np
from os import urandom



import des_for_linear as ciph
#only for 3-round linear expression of des
#Ph[7,18,24,29]+Pl[15]+Ch[7,18,24,29]+Cl[15]==k1[22]+k3[22]


#
def get_Round():
    return 3


def obtain_subkey_use_L3(p,c):
    pr=0.5+1.56*pow(2,-3)

    plain=p
    cipher=c
    
    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])
    

    #Guess_expression_key=0
    
    sample=Ph[:,31-7]^Ph[:,31-18]^Ph[:,31-24]^Ph[:,31-29]^Pl[:,31-15]^Ch[:,31-7]^Ch[:,31-18]^Ch[:,31-24]^Ch[:,31-29]^Cl[:,31-15]
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

def attack_3_round_use_L3(num_pair):

    
    #generate randomly a key
    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,get_Round())#the round of attack
    
    #generate sample_pair*pc_pair plaintext
    plain = np.frombuffer(urandom(64*num_pair), dtype=np.uint8).reshape(-1,64)
    plain = plain & 1

    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    #encrypt
    for sk in subkey:
        Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
    cipher=np.concatenate((Pl,Ph),axis=1)
    
    real_key=subkey[0][:,47-22]^subkey[2][:,47-22]
    real_key=real_key[0]
    
    
    guess_key=obtain_subkey_use_L3(plain,cipher)
    
    
    #print(real_key,end='----')
    #print(guess_key)
    
    return real_key,guess_key,
    

num=10**5
flag=0


num_pair=32

Real_key=[]
Guess_key=[]

NUM_pair=[int(num_pair)]

for num_pair in NUM_pair:

    for i in range(num): 
    
        real_key,guess_key=attack_3_round_use_L3(num_pair)
        Real_key.append(real_key)
        Guess_key.append(guess_key)
        np.save('real_key_for_mastui'+str(num_pair)+'.npy',np.array(Real_key,dtype=np.uint8))
        np.save('guess_key_for_mastui'+str(num_pair)+'.npy',np.array(Guess_key,dtype=np.uint8))

    
        if(real_key==guess_key):
            flag=flag+1
        print(i,flag/(i+1))

print(flag/num)
    
    
    