import numpy as np

def escalon(x):
    return np.heaviside(x,1)        # 0 if x<0 else 1

def sign(x):
    return np.sign(x)

def relu(x):    
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def sigmoide(x):
    return 1/(1 + np.exp(-x))