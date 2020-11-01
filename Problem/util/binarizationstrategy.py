#!/usr/bin/python
# encoding=utf8
"""
Created on Tue Apr 23 19:05:37 2019

@author: cavasquez
"""
import math
import random
import multiprocessing as mp
import numpy as np

class BinarizationStrategy:
    def __init__(self,tecnica,binary):
        self._b = []
#        self.x = x  
        self.tecnica = tecnica
        matrizbinaria = []
        
        
        self.funcShape = None
        if(self.tecnica=="sShape1"):
            self.funcShape = self.sShape1
        if(self.tecnica=="sShape2"):
            self.funcShape = self.sShape2
        if(self.tecnica=="sShape3"):
            self.funcShape = self.sShape3
        if(self.tecnica=="sShape4"):
            self.funcShape = self.sShape4
        if(self.tecnica=="vShape1"):
            self.funcShape = self.vShape1
        if(self.tecnica=="vShape2"):
            self.funcShape = self.vShape2
        if(self.tecnica=="vShape3"):
            self.funcShape = self.vShape3
        if(self.tecnica=="vShape4"):
            self.funcShape = self.vShape4
        
#        tb = [funcShape(item) for item in x]
        
#        pool = mp.Pool(4)
#        tb = pool.map(funcShape, x)
#        pool.close()
        
        self.funcBin = None
        if(binary=="Standar"):
            self.funcBin = self.standard
            
        if(binary=="invStandar"):
            self.funcBin = self.invStandard
        
        if(binary=="Complement"):
            self.funcBin = self.complement
        
        if (binary=="StaticProbability"):
            self.funcBin = self.staticProbability
            
        if(binary=="Elitist"):
            self.funcBin = self.elitist
            
        self.mejorSol = None
        
#        matrizbinaria = [funcBin(item) for item in tb]
        
#        pool = mp.Pool(4)
#        matrizbinaria = pool.map(funcBin, tb)
#        pool.close()
        
        
#        for i in range(len(x)):
#            tb = 0
#            if(self.tecnica=="sShape1"):
#                tb = self.sShape1(x[i])
#            if(self.tecnica=="sShape2"):
#                tb = self.sShape2(x[i])
#            if(self.tecnica=="sShape3"):
#                tb = self.sShape3(x[i])
#            if(self.tecnica=="sShape4"):
#                tb = self.sShape4(x[i])
#            if(self.tecnica=="vShape1"):
#                tb = self.vShape1(x[i])
#            if(self.tecnica=="vShape2"):
#                tb = self.vShape2(x[i])
#            if(self.tecnica=="vShape3"):
#                tb = self.vShape3(x[i])
#            if(self.tecnica=="vShape4"):
#                tb = self.vShape4(x[i])
#            
#            if(binary=="Standar"):
#                matrizbinaria.append(self.standard(tb))
#            
#            if(binary=="Complement"):
#                matrizbinaria.append(self.complement(tb))
#            
#            if (binary=="StaticProbability"):
#                matrizbinaria.append(self.staticProbability(tb, 0.4))
#                
#            if(binary=="Elitist"):
#                matrizbinaria.append(self.elitist(tb))
                
        self.set_binary(matrizbinaria)
        #print(matrizbinaria)
        
    def binarize(self, x):
        tb = [self.funcShape(item) for item in x]        
        matrizbinaria = [self.funcBin(item, tb) for item in range(len(tb))]
        return matrizbinaria
    
    def binarizeBatch(self, matriz, mejorSol):
        tb = np.vectorize(self.funcShape)(matriz)
        matrizbinaria = np.vectorize(self.funcBin)(tb)
    
        return matrizbinaria
    
    def set_binary(self,binary):
        self._b = binary
    
    def get_binary(self):
        return self._b
    
    def basic(self, x):
        if 0.5 <= x: 
           return 1 
        else:
           return 0
    
    def standard(self, idx, tb):
        if tb is None or idx is None: return 1
        if random.random() <= tb[idx]:
            return 1.
        else: 
            return 0.
        
    def invStandard(self, idx, tb):
        if tb is None or idx is None: return 1
        if random.random() >= tb[idx]:
            return 1.
        else: 
            return 0.
        
    def complement(self,x):
        if random.random() <= x:
            return  self.standard(1 -x)
        else: 
            return 0
    
    def staticProbability(self, x, alpha):
        if alpha >= x: 
            return 0 
        elif (alpha < x and x <= ((1 + alpha) / 2)):
            return self.standard(x)
        else: 
            return 1
        
    def elitist(self, idx, tb):
        
        if random.random() < tb[idx]:
            return self.mejorSol[idx] if self.mejorSol is not None else self.standard(idx,tb)
        else: 
            return self.standard(idx,tb)
    
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape1(self,x):
        
        try:
            return (1 / (1 + math.pow(math.e, -2 * x)))
        except OverflowError:
            print(f'x {x}\ne {math.e}')
            
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape2(self,x):
        return (1 / (1 + math.pow(math.e, -x)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape3(self,x):
        return (1 / (1 + math.pow(math.e, -x / 2)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def sShape4(self,x):
        return (1 / (1 + math.pow(math.e, -x / 3)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape1(self,x):
        return abs(self.erf((math.sqrt(math.pi) / 2) * x))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape2(self,x):
        return abs(math.tanh(x))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape3(self,x):
        return abs(x / math.sqrt(1 + math.pow(x, 2)))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def vShape4(self,x):
        return abs((2 / math.pi) * math.atan((math.pi / 2) * x))
    #-----------------------------------------------------------------------------------------------------------------------------   
    def erf(self,z):
        q = 1.0 / (1.0 + 0.5 * abs(z))
        ans = 1 - q * math.exp(-z * z - 1.26551223 + q * (1.00002368
                    + q * (0.37409196
                    + q * (0.09678418
                    + q * (-0.18628806
                    + q * (0.27886807
                    + q * (-1.13520398
                    + q * (1.48851587
                    + q * (-0.82215223
                    + q * (0.17087277))))))))))
        if z >= 0:
            return ans
        else:
            return -ans
    