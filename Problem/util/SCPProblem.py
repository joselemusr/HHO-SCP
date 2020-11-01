#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:02:44 2019

@author: mauri
"""
import numpy as np
from . import read_instance as r_instance
from . import binarizationstrategy as _binarization
from .repair import ReparaStrategy as _repara
from . import permutationRank as PermRank
from datetime import datetime
import multiprocessing as mp
from numpy.random import default_rng

class SCPProblem():
    def __init__(self, instancePath = None):
#        print(f'LEYENDO INSTANCIA')
        self.instancia = instancePath
        self.instance = r_instance.Read(instancePath)
        self.optimo = self.instance.optimo
#        print(f'FIN LEYENDO INSTANCIA')
        if(self.instance.columns != np.array(self.instance.get_c()).shape[0]):
            raise Exception(f'self.instance.columns {self.instance.columns} != np.array(self.instance.get_c()).shape[1] {np.array(self.instance.get_c()).shape[1]})')
        self.tTransferencia = "sShape2"
        self.tBinary = "Standar"        
        self.binarizationStrategy = _binarization.BinarizationStrategy(self.tTransferencia, self.tBinary)        
        self.repair = _repara.ReparaStrategy(self.instance.get_r()
                                            ,self.instance.get_c()
                                            ,self.instance.get_rows()
                                            ,self.instance.get_columns())
        self.paralelo = False
        self.penalizar = False
        self.mejorSolHist = np.ones((self.instance.get_columns())) * 0.5
        self.mejorFitness = None

        self.partSize = 8
        self.rangeMax = []
        self.permRank = PermRank.PermRank()
        self.particiones = []
        for _ in range(int(self.instance.get_columns()/self.partSize)):
            self.rangeMax.append(self.permRank.totalPerm(self.partSize))
            self.particiones.append(self.partSize)

        if self.instance.get_columns()%self.partSize > 0:
            self.rangeMax.append(self.permRank.totalPerm(self.instance.get_columns()%self.partSize))
            self.particiones.append(self.instance.get_columns()%self.partSize)
        self.rangeMax = np.array(self.rangeMax)
        self.particiones = np.array(self.particiones)


    def getNombre(self):
        return 'SCP'
    
    def getNumDim(self):
        #return self.instance.columns
        return self.particiones.shape[0]

    def getRangoSolucion(self):
        return {'max': self.rangeMax, 'min':np.zeros(self.rangeMax.shape[0])}

    def eval(self, encodedInstance):
        decoded, numReparaciones = self.frepara(encodedInstance)
        fitness = self.evalInstance(encodedInstance)
        return fitness, decoded, numReparaciones

    def evalEnc(self, encodedInstance):
        decoded, numReparaciones = self.decodeInstance(encodedInstance)
        fitness = self.evalInstance(decoded)
        if self.mejorFitness is None or fitness > self.mejorFitness:
            self.mejorFitness = fitness
            self.binarizationStrategy.mejorSol = decoded
        encoded = self.encodeInstance(decoded)
        return fitness, decoded, numReparaciones,encoded

    def evalEncBatch(self, encodedInstances, mejorSol):
        decoded, numReparaciones = self.decodeInstancesBatch(encodedInstances, mejorSol)
        fitness = self.evalInstanceBatch(decoded)
        
        
        return fitness, decoded, numReparaciones
    
    def evalDecBatch(self, encodedInstances, mejorSol):
        fitness = self.evalInstanceBatch(encodedInstances)
        
        
        return fitness, encodedInstances, None
    
    def encodeInstance(self, decodedInstance):
        currIdx = 0
        res = []
        for partSize in self.particiones:
            res.append(self.permRank.getRank(decodedInstance[currIdx:currIdx+partSize]))
            currIdx+=partSize
        return np.array(res)

#    @profile
        
    def decodeInstancesBatch(self, encodedInstances, mejorSol):
        start = datetime.now()
        b = self.binarizationStrategy.binarizeBatch(encodedInstances, mejorSol)
        end = datetime.now()
        binTime = end-start
        numReparaciones = 0
        repaired = self.repair.reparaBatch(b)
        return repaired, numReparaciones
    
    
    def decodeInstance(self, encodedInstance):
        encodedInstance = np.array(encodedInstance).astype(np.int8)
        if encodedInstance.shape[0] != self.particiones.shape[0]:
            raise Exception("La instancia encodeada cambio su tamaño")

        binario = []
        #print(encodedInstance)
        #raise Exception
        for idx in range(encodedInstance.shape[0]):
            #print(f"self.particiones[idx], encodedInstance[idx] {self.particiones[idx]}, {encodedInstance[idx]}")
            binario.extend(self.permRank.unrank(self.particiones[idx], encodedInstance[idx]).tolist())
        b = np.array(binario)
        

        #b = self.binarizationStrategy.binarize(encodedInstance)
        numReparaciones = 0
        if not self.penalizar:
                b, numReparaciones = self.frepara(b)
        return b, numReparaciones
        
    def binarize(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
   
#    @profile
    def evalInstance(self, decoded):
        return -(self.fObj(decoded, self.instance.get_c())) if self.repair.cumple(decoded) == 1 else -1000000
    
    def evalInstanceBatch(self, decoded):
        start = datetime.now()
        ret = np.sum(np.array(self.instance.get_c())*decoded, axis=1)
        end = datetime.now()
        return -ret
    
#    @profile
    def fObj(self, pos,costo):
        return np.sum(np.array(pos) * np.array(costo))
  
#    @profile
    def freparaBatch(self,x):
        start = datetime.now()
        #print(x.shape)
        exit()
        end = datetime.now()
    
    
    def frepara(self,x):
        start = datetime.now()
        cumpleTodas=0
        cumpleTodas=self.repair.cumple(x)
        if cumpleTodas == 1: return x, 0
        
        x, numReparaciones = self.repair.repara_one(x)    
        x = self.mejoraSolucion(x)
        end = datetime.now()
        return x, numReparaciones
    
    def mejoraSolucion(self, solucion):
        solucion = np.array(solucion)
        costos = solucion * self.instance.get_c()
        cosOrd = np.argsort(costos)[::-1]
        for pos in cosOrd:
            if costos[pos] == 0: break
            modificado = solucion.copy()
            modificado[pos] = 0
            if self.repair.cumple(modificado) == 1:
                solucion = modificado
        return solucion
    
    def generarSolsAlAzar(self, numSols, mejorSol=None):
#        args = []
        if mejorSol is None:
            args = np.zeros((numSols, self.getNumDim()), dtype=np.float)
        else:
            #self.mejorSolHist = (mejorSol+self.mejorSolHist)/2
            args = []
            for i in range(numSols):
                sol = mejorSol.copy()
                idx = np.random.randint(low=0, high=sol.shape[0])
                sol[idx] = np.random.randint(low=0, high=self.particiones[idx])
                args.append(sol)
            args = np.array(args)
        fitness = []
        ant = self.penalizar
        self.penalizar = False
        if self.paralelo:
            pool = mp.Pool(4)
            ret = pool.map(self.evalEnc, args.tolist())
            pool.close()
            fitness =  np.array([item[0] for item in ret])
            sol = np.array([item[3] for item in ret])
        else:
            sol = []
            for arg in args:
                sol_ = np.array(self.evalEnc(arg)[3])
                fitness_ = np.array(self.evalEnc(arg)[0])
                sol.append(sol_)
                fitness.append(fitness_)
            sol = np.array(sol)
            fitness = np.array(fitness)
        self.penalizar = ant

        return sol, fitness
    
    def graficarSol(self, datosNivel, parametros, nivel, id = 0):
        if not hasattr(self, 'graficador'):
            self.initGrafico()
        y = datosNivel['soluciones'][0]
        vels = datosNivel['velocidades'][0]
        self.graficador.live_plotter(np.arange(y.shape[0]),y, 'soluciones', dotSize=0.1, marker='.')
        self.graficador.live_plotter(np.arange(vels.shape[0]), vels, 'velocidades', dotSize=0.1, marker='.')
        self.graficador.live_plotter(np.arange(parametros.shape[0]), parametros, 'paramVel', dotSize=1.5, marker='-')
        