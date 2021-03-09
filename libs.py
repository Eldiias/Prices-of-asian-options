# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 23:36:31 2018

@author: eldiy
"""

import numpy as np
from random import gauss
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import axes3d, Axes3D

#Declaration of the parameters
T=1
r=0.03
sigma=0.2
x0=100
K1=100
K2=110
m=50
n=10000

def X(x0,t,b,r,s):
    y=x0*np.exp(((r-0.5*s**2)*t)+s*b)
    return (y)

def dig(K1,K2,x):
    return int(K1<x<K2)


def prix(n,m,x,T,k1,k2,r,s):
    y=0
    pr_call=0
    var_call=0
    pr_dig=0
    var_dig=0
    for i in range(n):
        b=0
        for j in range(m):
            db=gauss(0.0,(T/m)**0.5)
            b += db
            x_s=X(x,T*j/m,b,r,s)
            y +=x_s
        y *= T/m
        pr_call += max(y-k1,0)
        var_call += max(y-k1,0)*max(y-k1,0)
        
        pr_dig += dig(k1,k2,y)
        var_dig += dig(k1,k2,y)*dig(k1,k2,y)
    var_call = ((var_call/n - (pr_call/n)**2))*np.exp(-2*r*T)/n 
    pr_call *= np.exp(-r*T)/n
    var_dig = ((var_dig/n - (pr_dig/n)**2))*np.exp(-2*r*T)/n 
    pr_dig *=  np.exp(-r*T)/n
    res=[pr_call,pr_dig, var_call, var_dig]
    return res

price=prix(n,m,x0,T,K1,K2,r,sigma)
print(price)


#study the convergence with N
m=200  #on fixe M
n=1000
N=51000
step=2000
fin=int((N-n)/step+1)
Res=[]
price_call=[]
price_dig=[]
var1=[]
var2=[]
kg=[]
up_bound_call=[]
low_bound_call=[]
up_bound_dig=[]
low_bound_dig=[]
for i in range(n,N,step):
    kg.append(i)
    Res=prix(n,m,x0,T,K1,K2,r,sigma)
    price_call.append(Res[0])
    price_dig.append(Res[1])
    up_bound_call.append(Res[0]+1.96*Res[2])
    low_bound_call.append(Res[0]-1.96*Res[2])
    up_bound_dig.append(Res[1]+1.96*Res[3])
    low_bound_dig.append(Res[1]-1.96*Res[3])
plt.figure(1)
plt.plot(kg,price_call)
plt.plot(kg,up_bound_call)
plt.plot(kg,low_bound_call)
plt.show

plt.figure(2)
plt.plot(kg,price_dig)
plt.plot(kg,up_bound_dig)
plt.plot(kg,low_bound_dig)

plt.show

#calculation of delta

def delta(n,m,x,T,k1,k2,r,s,eps):
    y1=0
    y2=0
    delta_call=0
    var_delta_call=0
    delta_dig=0
    var_delta_dig=0
    for i in range(n):
        b=0
        for j in range(m):
            db=gauss(0.0,(T/m)**0.5)
            b += db
            x_s_1=X(x+eps,T*j/m,b,r,s)
            x_s_2=X(x-eps,T*j/m,b,r,s)
            y1 +=x_s_1
            y2 +=x_s_2
        y1 *= T/m
        y2 *= T/m
        
        delta_call += max(y1-k1,0)-max(y2-k1,0)
        var_delta_call += (max(y1-k1,0)-max(y2-k1,0))*(max(y1-k1,0)-max(y2-k1,0))
        
        delta_dig += dig(k1,k2,y1)-dig(k1,k2,y2)
        var_delta_dig += (dig(k1,k2,y1)-dig(k1,k2,y2))*(dig(k1,k2,y1)-dig(k1,k2,y2))
    var_delta_call = ((var_delta_call/n - (delta_call/n)**2))*np.exp(-2*r*T)/(4*n*eps**2)
    delta_call *= np.exp(-r*T)/(2*n*eps)
    var_delta_dig = ((var_delta_dig/n - (delta_dig/n)**2))*np.exp(-2*r*T)/(4*n*eps**2)
    delta_dig *=  np.exp(-r*T)/(2*n*eps)
    res=[delta_call,delta_dig, var_delta_call, var_delta_dig]
    return res

#example
n=10000
m=200
eps=0.1

DRes=delta(n,m,x0,T,K1,K2,r,sigma,eps)

#study the convergence with eps
delta_call=[]
delta_dig=[]
up_bound_call=[]
low_bound_call=[]
up_bound_dig=[]
low_bound_dig=[]
kg=[]
for e in range(1,100):
    kg.append(e/10)
    Res=delta(n,m,x0,T,K1,K2,r,sigma,e/10)
    delta_call.append(Res[0])
    delta_dig.append(Res[1])
    up_bound_call.append(Res[0]+1.96*Res[2])
    low_bound_call.append(Res[0]-1.96*Res[2])
    up_bound_dig.append(Res[1]+1.96*Res[3])
    low_bound_dig.append(Res[1]-1.96*Res[3])
    
plt.figure(1)
plt.plot(kg,delta_call)
plt.plot(kg,up_bound_call)
plt.plot(kg,low_bound_call)
plt.show

plt.figure(2)
plt.plot(kg,delta_dig)
plt.plot(kg,up_bound_dig)
plt.plot(kg,low_bound_dig)

plt.show


def Pi1(n,m,x,T,k1,k2,r,s):
    y=0
    y1=0
    y2=0
    pr_call=0
    var_call=0
    pr_dig=0
    var_dig=0
    delta1_call=0
    var_delta1_call=0
    delta1_dig=0
    var_delta1_dig=0
    for i in range(n):
        b=0
        for j in range(m):
            db=gauss(0.0,(T/m)**0.5)
            b += db
            x_s=X(x,T*j/m,b,r,s)
            y +=x_s
            y1 +=x_s*j
            y2 +=x_s*j*j
        y *= T/m
        y1 *= T/m
        y2 *= T/m
        delta1_call += max(y-k1,0)
        var_delta1_call += max(y-k1,0)*max(y-k1,0)
        
        delta1_dig += dig(k1,k2,y)
        var_delta1_dig += dig(k1,k2,y)*dig(k1,k2,y)
        
    Pi1=y/y1*(b/s+y2/(x*y1))
        
    var_delta1_call = ((var_delta1_call/n - (delta1_call/n)**2))*np.exp(-2*r*T)/n 
    delta1_call *= np.exp(-r*T)/n*Pi1
    var_delta1_dig = ((var_delta1_dig/n - (delta1_dig/n)**2))*np.exp(-2*r*T)/n 
    delta1_dig *=  np.exp(-r*T)/n*Pi1
    res=[delta1_call,delta1_dig, var_delta1_call, var_delta1_dig]
    return res
        
Pi1(n,m,x0,T,K1,K2,r,sigma)

def Pi2(n,m,x,T,k1,k2,r,s):
    y=0
    y1=0
    y2=0
    delta1_call=0
    var_delta1_call=0
    delta1_dig=0
    var_delta1_dig=0
    for i in range(n):
        b=0
        for j in range(m):
            db=gauss(0.0,(T/m)**0.5)
            b += db
            x_s=X(x,T*j/m,b,r,s)
            y +=x_s
        y1 += x_s
        y *= T/m
        delta1_call += max(y-k1,0)
        var_delta1_call += max(y-k1,0)*max(y-k1,0)
        
        delta1_dig += dig(k1,k2,y)
        var_delta1_dig += dig(k1,k2,y)*dig(k1,k2,y)
        
    y1 *= 1/n
    Pi2=(2/(s*s)*((y1-x)/y -r))+1
        
    var_delta1_call = ((var_delta1_call/n - (delta1_call/n)**2))*np.exp(-2*r*T)/n 
    delta1_call *= np.exp(-r*T)/n*Pi2
    var_delta1_dig = ((var_delta1_dig/n - (delta1_dig/n)**2))*np.exp(-2*r*T)/n 
    delta1_dig *=  np.exp(-r*T)/n*Pi2
    res=[delta1_call,delta1_dig, var_delta1_call, var_delta1_dig]
    return res
        
Pi2(n,m,x0,T,K1,K2,r,sigma)

