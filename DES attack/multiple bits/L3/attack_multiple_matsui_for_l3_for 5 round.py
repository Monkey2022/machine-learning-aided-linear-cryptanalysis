# -*- coding: utf-8 -*-


import numpy as np
from os import urandom


import des_for_linear as ciph
#only for 1+3+1-round linear expression of des

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

#使用的线性等式的轮数
def get_Round():
    return 3


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

def obtain_subkey_use_L3(p,c):
    
    s_box=S_box()
    
    num_pair=len(p)
    
    plain=p
    cipher=c
    
    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])
    
    score_k=[]

    
    #猜测
    for k1 in range(2**6):
        Guess_k1=num2bitarray6(k1)
        Pl_extend=Pl[:,[31,0,1,2,3,4]]#
        
        for j in range(len(Pl_extend)):
            Pl_extend[j]=Pl_extend[j]^Guess_k1
        buffer_k1=bit2num(Pl_extend)
        buffer_k1=s_box[0][buffer_k1]
        
        ahead_part=Ph[:,31-15]^buffer_k1[:,1]^Pl[:,31-7]^Pl[:,31-18]^Pl[:,31-24]^Pl[:,31-29]
        
        for k5 in range(2**6):
            Guess_k5=num2bitarray6(k5)
            Cl_extend=Cl[:,[31,0,1,2,3,4]]#
            
            for j in range(len(Cl_extend)):
                Cl_extend[j]=Cl_extend[j]^Guess_k5
            buffer_k5=bit2num(Cl_extend)
            buffer_k5=s_box[0][buffer_k5]
            
            behind_part=Cl[:,31-7]^Cl[:,31-18]^Cl[:,31-24]^Cl[:,31-29]^Ch[:,31-15]^buffer_k5[:,1]
            
            sample=behind_part^ahead_part
            
            sample=np.array(sample,dtype=np.uint32)

            score_k.append(len(sample)-sum(sample))

    pr=0.5+1.56*pow(2,-3)
    max_score=max(np.array(score_k))
    min_score=min(np.array(score_k))
    
    if (abs(max_score-num_pair*0.5)>abs(min_score-num_pair*0.5)):
        guess_key=int(np.where(np.array(score_k)==max_score)[0][0])
        
        if(pr>0.5):
            guess_key=(0<<12)+guess_key
        else:
            guess_key=(1<<12)+guess_key
    else:
        guess_key=int(np.where(np.array(score_k)==min_score)[0][0])
        if(pr>0.5):
            guess_key=(1<<12)+guess_key
        else:
            guess_key=(0<<12)+guess_key
    
    
    guess_key=bin(guess_key)[2:]
    guess_key=guess_key.zfill(13)
    guess_key=list(guess_key)
    guess_key=np.array(guess_key,dtype=np.uint8)
    
    return guess_key


def attack_5_round_use_L3(num_pair):

    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,1+get_Round()+1)#
    plain = np.frombuffer(urandom(64*num_pair), dtype=np.uint8).reshape(-1,64)
    plain = plain & 1

    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    
    for sk in subkey:
        Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
    cipher=np.concatenate((Pl,Ph),axis=1)#
    
    real_key=np.zeros(13,dtype=np.uint8)
    
    real_key[0]=subkey[1][0,47-22]^subkey[3][0,47-22]
    real_key[1]=subkey[0][0,0]
    real_key[2]=subkey[0][0,1]
    real_key[3]=subkey[0][0,2]
    real_key[4]=subkey[0][0,3]
    real_key[5]=subkey[0][0,4]
    real_key[6]=subkey[0][0,5]
    
    real_key[7]=subkey[4][0,0]
    real_key[8]=subkey[4][0,1]
    real_key[9]=subkey[4][0,2]
    real_key[10]=subkey[4][0,3]
    real_key[11]=subkey[4][0,4]
    real_key[12]=subkey[4][0,5]
    
    guess_key=obtain_subkey_use_L3(plain,cipher)
    

    
    return real_key,guess_key
    
    
num=1000
flag=0
num_pair=16*10

for i in range(num):   
    real_key,guess_key=attack_5_round_use_L3(num_pair)

    r=array2num(real_key)
    g=array2num(guess_key)
    if(r==g):
        flag=flag+1
    print(i,num_pair,flag/(i+1))

print(flag/num)

    
    