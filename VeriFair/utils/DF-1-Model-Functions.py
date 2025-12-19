#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# reinterpret network symbolically using z3 variables.
import sys
from z3 import *
import numpy as np 
import pandas as pd 
import collections
import time
import datetime
from utils.verif_utils import *

def ground_net(x):
    layer_outs = []
    for i in range(len(w)):
        layer = []
        for j in range(len(w[i][0])):
            sum = 0
            for k in range(len(x)):
                sum += x[k] * w[i][k][j]
            sum += b[i][j]
        layer.append(sum)
        layer = np.asarray(layer, dtype=np.float64)
        y = layer if i == len(w)-1 else relu(layer)
        layer_outs.append(y)
        x = y
    return y

def layer_net(x, w, b):
    layers = []    
    for i in range(len(w)):
        x1 = w[i].T @ x + b[i]
        y1 = x1 if i == len(w)-1 else relu(x1)
        layers.append(y1)
        x = y1
    return layers

def net(x, w, b):
    x1 = w[0].T @ x + b[0]
    y1 = relu(x1)
    
    x2 = w[1].T @ y1 + b[1]
    y2 = relu(x2)
    
    x3 = w[2].T @ y2 + b[2]
    
    return x3

def z3_net(x, w, b):
    # Don't create unused Float32 variables
    # Just convert Int to Real directly
    fl_x = np.array([ToReal(x[i]) for i in range(len(x))])
    
    # Layer 1: 23 -> 16
    x1 = w[0].T @ fl_x + b[0]
    y1 = z3Relu(x1)
    
    # Layer 2: 16 -> 8
    x2 = w[1].T @ y1 + b[1]
    y2 = z3Relu(x2)
    
    # Layer 3: 8 -> 1 (output layer, no activation)
    x3 = w[2].T @ y2 + b[2]
    
    # Return scalar
    return x3[0] if hasattr(x3, '__getitem__') else x3