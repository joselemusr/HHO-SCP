#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:00:31 2019

@author: mauri
"""

#import readOrProblems as rOP
#from . import solution as sl
#from . import heuristic as he
#from . import matrixUtility as mu
import numpy as np
from datetime import datetime
#import cupy as cp
import math

class ReparaStrategy:
    
    def __init__(self, matrix, pesos, row, cols):
#        matrix = np.array(matrix)
        self.rows = row
        self.cols = cols
        self.pesos = np.array(pesos)
        self.matrix = np.array(matrix)
        sumaFilasRestriccion = np.sum(self.matrix, axis=1)
#        print(self.matrix)
#        print(self.matrix.shape)
#        print(sumaFilasRestriccion)
#        print(sumaFilasRestriccion.shape)
#        exit()
        ponderacion = (self.matrix.T * sumaFilasRestriccion).astype('float32')
        
        ponderacion = ponderacion.T
        
        ponderacion = np.sum(ponderacion, axis=0)
#        print(f'ponderacion {ponderacion}')
#        print(f'pesos {self.pesos}')
        
        self.ponderacion = ponderacion/(self.pesos*self.pesos)
        
#        print(f'ponderacion {self.ponderacion}')
#        exit()


    def cumple(self, solucion):
        #exit()
        incumplidas = self.cumpleBatch(np.array([solucion]))
        return 1 if len(np.where(incumplidas == 0)[0]) == 0 else 0

    def cumpleBatch(self, soluciones):
        
        soluciones = soluciones.astype('uint8')
        
        
        rexpanded = self.matrix.copy().astype('uint8')
        rexpanded = np.repeat(rexpanded, soluciones.shape[0], axis=0).reshape((self.matrix.shape[0],soluciones.shape[0],self.matrix.shape[1]))
        #print(rexpanded.shape)
        #print(rexpanded)
        mult = rexpanded*soluciones
#        if(np.sum(soluciones[0])>0):
#            print('***************************************************')
#            print(f'rexpanded {rexpanded.shape}')
#            print(f'soluciones {soluciones.shape}')
#            print(f'mult {mult.shape}')
##            print(self.matrix.astype('uint8'))
#            exit()
#        print(mult.shape)
#        print(mult)
#        blocks = int(math.ceil(mult.shape[0]/10))
#        res = []
#        print(rexpanded)
        suma = np.sum(mult, axis=2, dtype='float64').T
#        print(suma)
#        exit()
        suma[suma>0]  = 1
#        print(suma)
        producto = np.prod(suma, axis=1)
#        print(producto)
#        print((producto > 0))
        return np.where(producto == 0)[0]#.astype('uint8')

    def reparaBatch(self, soluciones):
        gstart = datetime.now()
#        pesos = self.pesos
        ponderacion = self.ponderacion
#        s = datetime.now()
#        pesosAplicados = soluciones * pesos
#        e = datetime.now()
#        s = datetime.now()
        #print(f'aplicacion pesos {e-s}')
#        sumaPesos = np.sum(pesosAplicados, axis=1)
#        e = datetime.now()
        #print(f'suma pesos {e-s}')
        #print(sumaPesos)
#        incumplidas = []
#        for idx in range(soluciones.shape[0]):
#            if not self.cumple(soluciones[idx]):
#                incumplidas.append(idx)
#                
#        incumplidas = np.array(incumplidas)
        
        incumplidas = self.cumpleBatch(soluciones)
        

#        print(incumplidas)
#        exit()
#        incumplida = np.where(sumaPesos > capacidad)
        #print(len(incumplidas[0]))
        #exit()
        negativo = soluciones.copy()
        negativo = np.where(negativo==0, 2, negativo)
        negativo = np.where(negativo==1, 0, negativo)
        negativo = np.where(negativo==2, 1, negativo)
        while (incumplidas == 0).any():
            start = datetime.now()
#            print(len(incumplidas))
#            print(np.sum(soluciones))
#            print(soluciones[0])
#            print(incumplidas)
            #print(f'incumplidas {incumplidas}')
            #exit()
#                s = datetime.now()
            
#            print(negativo)
#            exit()
#            negativoSols = soluciones[incumplidas]
            solucionesPonderadas = negativo[incumplidas] * ponderacion
            
            blocks = 30
            expanded = solucionesPonderadas.copy()
            expanded = np.repeat(expanded, blocks, axis=0)
#                e = datetime.now()
            #print(f'ponderacion columnas {e-s}')
            #print(solucionesPonderadas)
#            print(negativo.shape)
#            exit()
            k = int(negativo.shape[1]*0.1) #numero de candidatos eliminar
#            k=40
#            k = 10
            l = int(k*0.05) #numero de columnas a eliminar
#            l=10
#            print(f'l {l}')
#            print(f'k {k}')
#            exit()
            
            l = l if l > 0 else 1
#            l=1
            plantilla = np.ones((expanded.shape[0], l)) * -1
            
#            for i in range(blocks):
#                plantilla[i*blocks:(i+1)*blocks] = np.random.randint(l, size=(peoresIndices.shape[0],l))
#                s = datetime.now()
#            print(f'solucionesPonderadas {solucionesPonderadas}')
            
            peoresIndices = np.argpartition(-solucionesPonderadas,k,axis=1)[:,l-1::-1]
#            peoresIndices = np.argpartition(-solucionesPonderadas,k,axis=1)[:,k-1::-1]
            
            
#                e = datetime.now()
            peoresIndicesEliminar = np.random.randint(peoresIndices.shape[1], size=(peoresIndices.shape[0],l))
#            print(f'peoresIndicesEliminar {peoresIndicesEliminar}')
#            print(f'peoresIndicesEliminar {peoresIndicesEliminar.shape}')
#            exit()
#            print(peoresIndicesEliminar)
#            print(peoresIndicesEliminar.shape)
#            exit()
            #print(peoresIndices)
            #print(np.arange(len(peoresIndices)).reshape((-1,1)))
            #exit()
#                s = datetime.now()
            indicesEliminar = peoresIndices[np.arange(len(peoresIndices)).reshape((-1,1)), peoresIndicesEliminar]
            
            
            filas = indicesEliminar.shape[0]
#            l = 10
#            blocks = 3
            plantilla = np.ones((filas*blocks, l)) * -1
#            print(f'plantilla.shape {plantilla.shape}')
            random = indicesEliminar.copy()
#            print(f'random.shape {random.shape}')
            for i in range(blocks):
                plantilla[i*filas:(i+1)*filas, :int(math.ceil((i+1)*l/blocks))] = random[:, :int(math.ceil((i+1)*l/blocks))]
                plantilla[i*filas:(i+1)*filas, int(math.ceil((i+1)*l/blocks)):] = random[:, 0].reshape(-1,1)
            
#            print(f'plantilla {plantilla}')
#            print(f'plantilla {plantilla.shape}')
#            exit()
            
#            print(indicesEliminar)
#            print(indicesEliminar.shape)
#            exit()
            idxIncumplidasExpanded = np.repeat(incumplidas.reshape((1,incumplidas.shape[0])), blocks, axis=0)
#            print(idxIncumplidasExpanded)
#            print(np.arange(idxIncumplidasExpanded.shape[0]*idxIncumplidasExpanded.shape[1]).reshape(-1,1))
#            exit()
#            print(idxIncumplidasExpanded.reshape(-1,1).shape)
##            exit()
            
            
            solCumple = soluciones.copy()
#            print(solCumple)
            solCumple = np.repeat(solCumple, blocks, axis=0)
            solCumple[np.arange(idxIncumplidasExpanded.shape[0]*idxIncumplidasExpanded.shape[1]).reshape(-1,1),plantilla.astype('int32')] = 1
#            print(solCumple)
#            print(f'solcumple {solCumple.shape}')
#            exit()
#            print(idxIncumplidasExpanded.shape)
#            exit()
            incumplidasExpanded = np.ones((solCumple.shape[0]))
#            resCumple = self.cumpleBatch(solCumple)
#            print(resCumple)
#            exit()
            incumplidasExpanded[self.cumpleBatch(solCumple)] = 0
#            idx
#            print(incumplidasExpanded)
#            print(incumplidasExpanded.shape)
#            exit()
#            print(incumplidasExpanded.shape)
#            exit()
            incumplidasExpanded = incumplidasExpanded.reshape( (blocks, soluciones.shape[0], 1) )
#            print(incumplidasExpanded)
#            print(incumplidasExpanded.shape)
#            exit()
            
            elegidos = np.argmax(incumplidasExpanded, axis=0)
            elegidos = elegidos.reshape((elegidos.shape[0]))
#            print(elegidos)
#            print(elegidos.shape)
#            
#            exit()
            cumpleElegidos = solCumple.copy()
            cumpleElegidos = cumpleElegidos.reshape( (blocks, soluciones.shape[0], cumpleElegidos.shape[1]) )
            soluciones = cumpleElegidos[elegidos,np.arange(soluciones.shape[0]),:]
#            soluciones = cumpleElegidos[elegidos,np.arange(soluciones.shape[0]),:]
#            print(f'soluciones {(soluciones == solNuevo).all()}')
#            exit()
#            print(f'cumpleElegidos {cumpleElegidos}')
#            print(f'cumpleElegidos {cumpleElegidos.shape}')
#            exit()
#            print(f'incumplidasExpanded {incumplidasExpanded.shape}')
#            print(f'incumplidasExpanded {incumplidasExpanded}')
#            exit()
            incumplidas = incumplidasExpanded[elegidos,np.arange(soluciones.shape[0]),:].reshape(soluciones.shape[0]).astype('uint8')
#            print(incumplidas)
#            print(incumplidas.shape)
#            exit()
            
            
            
            
#            solsCumplidas = np.ones((solCumple.shape[0]))
#            solsCumplidas[incumplidasExpanded] = 0
##            print(solsCumplidas)
##            print(solsCumplidas.shape)
##            exit()
#            solsCumplidas = solsCumplidas.reshape( (blocks, soluciones.shape[0], 1) )
##            print(solsCumplidas)
##            print(solsCumplidas.shape)
#            elegidos = np.argmax(solsCumplidas, axis=0)
#            print(elegidos)
#            print(elegidos.shape)
#            elegidos = elegidos.reshape((elegidos.shape[0]))
#            print(elegidos)
#            print(elegidos.shape)
#            exit()
#            
#            #indicesEliminar = peoresIndices[:, peoresIndicesEliminar]
##                e = datetime.now()
##            print(f'indicesEliminar {indicesEliminar}')
##            print(f'indicesEliminar {indicesEliminar.shape}')
#            #print(np.array(incumplidas[0]).reshape((-1,1)))
##            exit()
#
#            soluciones[np.array(incumplidas[0]).reshape((-1,1)),indicesEliminar] = 1
            negativo = soluciones.copy()
            negativo = np.where(negativo==0, 2, negativo)
            negativo = np.where(negativo==1, 0, negativo)
            negativo = np.where(negativo==2, 1, negativo)
#                s = datetime.now()
#                pesosAplicados = soluciones * pesos
#                e = datetime.now()
            #print(f'pesosAplicados {e-s}')
            #print(f'pesosAplicados {pesosAplicados}')
#                s = datetime.now()
#                sumaPesos = np.sum(pesosAplicados, axis=1)
#                e = datetime.now()
            #print(f'sumaPesos {e-s}')
            #print(sumaPesos)
#                incumplidas = np.where(sumaPesos > capacidad)
#            incumplidas = []
#            for idx in range(soluciones.shape[0]):
#                if not self.cumple(soluciones[idx]):
#                    incumplidas.append(idx)
#            incumplidas = np.array(incumplidas)
            end = datetime.now()
#            print(f'resto de la iteracion demora {end-start}')
#            start = datetime.now()
#            incumplidas = self.cumpleBatch(soluciones)
#            end = datetime.now()
#            print(f'verificar cumplimiento restricciones demoro {end-start}')
        end = datetime.now()
        print(f'reparaBatch demoro {end-gstart}')
#        print(soluciones)
#        for i in range(soluciones.shape[0]-1):
#            print(f'soluciones[{i}] { soluciones[i] } == soluciones[{i+1}] {soluciones[i] } { (soluciones[i] == soluciones[i+1]).all() }' )
        return soluciones