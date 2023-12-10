# -*- coding: utf-8 -*-



import numpy as np
from os import urandom

import des_for_linear as ciph
#only for 6-round linear expression of des



#使用的线性等式的轮数
def get_Round():
    return 6


def obtain_subkey_use_L6(p,c):
    

    pr=0.5-1.95*pow(2,-9)
    
    plain=p
    cipher=c
    
    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])

    #Guess_expression_key=0
    
    sample=Pl[:,31-7]^Pl[:,31-18]^Pl[:,31-24]^Ch[:,31-7]^Ch[:,31-18]^Ch[:,31-24]^Ch[:,31-29]^Cl[:,31-15]
    
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
    
 


def attack_6_round_use_L6(num_pair):

    

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
    
    real_key=subkey[1][:,47-22]^subkey[2][:,47-44]^subkey[3][:,47-22]^subkey[5][:,47-22]
    real_key=real_key[0]
    
    
    guess_key=obtain_subkey_use_L6(plain,cipher)

    
    return real_key,guess_key
    

num=10**5
flag=0


num_pair=2048

Real_key=[]
Guess_key=[]

NUM_pair=[int(num_pair)]

for num_pair in NUM_pair:

    for i in range(num): 
    
        real_key,guess_key=attack_6_round_use_L6(num_pair)
        Real_key.append(real_key)
        Guess_key.append(guess_key)
        np.save('real_key_for_mastui'+str(num_pair)+'.npy',np.array(Real_key,dtype=np.uint8))
        np.save('guess_key_for_mastui'+str(num_pair)+'.npy',np.array(Guess_key,dtype=np.uint8))

    
        if(real_key==guess_key):
            flag=flag+1
        print(i,flag/(i+1))

print(flag/num)
    
    
    