# Utils

import sys
import os
import settings
from envs import env
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import math

# SQL
import sqlalchemy as db
import psycopg2
import json
import pickle
import zlib

import Database.Database as Database
# MH

from Problem.util import read_instance as Instance
from Problem import SCP as Problem
from Metrics import Diversidad as dv

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+env('DIR_INSTANCES')

connect = Database.Database()


def HHO_SCP(id,instance_file,instance_dir,population,maxIter,discretizacionScheme):

    instance_path = workdirInstance + instance_dir + instance_file

    if not os.path.exists(instance_path):
        print(f'No se encontró la instancia: {instance_path}')
        return False

    instance = Instance.Read(instance_path)

    matrizCobertura = np.array(instance.get_r())
    vectorCostos = np.array(instance.get_c())

    dim = len(vectorCostos)
    pob = population
    maxIter = maxIter
    DS = discretizacionScheme #[v1,Standard]
    a = 2

    #Variables de diversidad
    diversidades = []
    maxDiversidades = np.zeros(7) #es tamaño 7 porque calculamos 7 diversidades
    PorcentajeExplor = []
    PorcentajeExplot = []
    state = []

    #Generar población inicial
    poblacion = np.random.uniform(low=-1.0, high=1.0, size=(pob,dim))
    matrixBin = np.zeros((pob,dim))
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)
    matrixBin,fitness,solutionsRanking  = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)

    timerStart = time.time()
    timerStartResult = time.time()
    memory = []

    #Parámetros fijos de HHO
    beta=1.5 #Escalar según paper
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) #Escalar
    LB = -10 #Limite inferior de los valores continuos
    UB = 10 #Limite superior de los valores continuos


    for iter in range(0, maxIter):
        processTime = time.process_time()  

        if iter == 0:
            if not connect.startEjecucion(id,datetime.now(),'ejecutando'):
                return False
           

        timerStart = time.time()
        
        #HHO
        E0 = np.random.uniform(low=-1.0,high=1.0,size=pob) #vector de tam Pob
        E = 2 * E0 * (1-(iter/maxIter)) #vector de tam Pob
        Eabs = np.abs(E)
        
        q = np.random.uniform(low=0.0,high=1.0,size=pob) #vector de tam Pob
        r = np.random.uniform(low=0.0,high=1.0,size=pob) #vector de tam Pob
        
        Xm = np.mean(poblacion,axis=0)

        bestRowAux = solutionsRanking[0]
        Best = poblacion[bestRowAux]
        BestBinary = matrixBin[bestRowAux]
        BestFitness = np.min(fitness)

        if np.min(Eabs) >= 1:
            if np.min(q) >= 0.5: # ecu 1.1
                indexCond11 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.1
                Xrand = poblacion[np.random.randint(low=0, high=pob, size=indexCond11.shape[0])] #Me entrega un conjunto de soluciones rand de tam indexCond11.shape[0] (osea los que cumplen la cond11)
                poblacion[indexCond11] = Xrand - np.multiply(np.random.uniform(low= 0.0, high=1.0, size=indexCond11.shape[0]),np.abs(Xrand-(2*np.multiply(np.random.uniform(low= 0.0, high=1.0, size = indexCond11.shape[0]),poblacion[indexCond11])))) #Aplico la ecu 1.1 solamente a las que cumplen las condiciones np.argwhere(Eabs>=1),np.argwhere(q>=0.5)

            else: # ecu 1.2
                indexCond12 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q<0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.2 
                poblacion[indexCond12] = (Best - Xm)- np.multiply( np.random.uniform(low= 0.0, high=1.0, size = indexCond12.shape[0]), (LB + np.random.uniform(low= 0.0, high=1.0, size = indexCond12.shape[0]) * (UB-LB)) )
        else:
            if np.min(Eabs) >= 0.5:
                if np.min(r) >= 0.5: # ecu 4
                    indexCond4 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 4
                    poblacion[indexCond4] = (Best - poblacion[indexCond4]) - np.multiply( E[indexCond4], np.abs( np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond4.shape[0])), Best )- poblacion[indexCond4] ) )                
                else: #ecu 10
                    indexCond10 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10
                    #ecu 7
                    y10 = poblacion
                    y10[indexCond10] = y10[bestRowAux]- np.multiply( E[indexCond10], np.abs( np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond10.shape[0])), y10[bestRowAux] )- y10[indexCond10] ) )  

                    #ecu 8
                    z10 = y10
                    S = np.random.uniform(low= 0.0, high=1.0, size=(y10.shape))
                    LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y10.shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y10.shape))),(1/beta)))
                    z10[indexCond10] = y10[indexCond10] + S[indexCond10]*LF

                    #evaluar fitness de ecu 7 y 8
                    Fy10 = solutionsRanking
                    Fy10[indexCond10] = Problem.SCP(y10[indexCond10],matrixBin[indexCond10],solutionsRanking[indexCond10],vectorCostos,matrizCobertura,DS)
                    
                    Fz10 = solutionsRanking
                    Fz10[indexCond10] = Problem.SCP(z10[indexCond10],matrixBin[indexCond10],solutionsRanking[indexCond10],vectorCostos,matrizCobertura,DS)
                    
                    #ecu 10.1
                    indexCond101 = np.intersect1d(indexCond10, np.argwhere(Fy10 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.1
                    poblacion[indexCond101] = y10[indexCond101]

                    #ecu 10.2
                    indexCond102 = np.intersect1d(indexCond10, np.argwhere(Fz10 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.2
                    poblacion[indexCond102] = z10[indexCond102]
            else:
                if np.min(r) >= 0.5: # ecu 6
                    indexCond6 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 6
                    poblacion[indexCond6] = Best- np.multiply(E[indexCond6], np.abs(Best - poblacion[indexCond6] ) )

                else: #ecu 11
                    indexCond11 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11
                    #ecu 12
                    y11 = poblacion
                    array_Xm = np.zeros(poblacion[indexCond11].shape)
                    array_Xm = array_Xm + Xm

                    #*** Cambiar y probar por:
                    #array_E = np.ones(poblacion[indexCond11].shape)
                    #array_E = array_E[indexCond11]*E[indexCond11]

                    array_E = np.zeros(poblacion[indexCond11].shape)
                    array_E[:,0] = E[indexCond11]
                    array_E[:,1] = E[indexCond11]

                    y11[indexCond11] = y11[bestRowAux]-  np.multiply(  array_E,  np.abs(  np.multiply(  2*(1-np.random.uniform(low= 0.0, high=1.0, size=poblacion[indexCond11].shape)),  y11[bestRowAux]  )- array_Xm ) )

                    #ecu 13
                    z11 = y11
                    S = np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))
                    LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y11.shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))),(1/beta)))
                    z11[indexCond11] = y11[indexCond11] + np.multiply(S[indexCond11],LF[[indexCond11]])

                    #evaluar fitness de ecu 12 y 13
                    if solutionsRanking is None: solutionsRanking = np.ones(pob)*999999
                    Fy11 = solutionsRanking
                    for i in indexCond11:
                        Fy11[i] = Problem.SCP(np.array(y11[i]),matrixBin[i],solutionsRanking[i],vectorCostos,matrizCobertura,DS)
                    Fz11 = solutionsRanking
                    for i in indexCond11:
                        Fz11[i] = Problem.SCP(z11[i],matrixBin[i],solutionsRanking[i],vectorCostos,matrizCobertura,DS)
                    
                    #ecu 11.1
                    indexCond111 = np.intersect1d(indexCond11, np.argwhere(Fy11 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.1
                    poblacion[indexCond111] = y11[indexCond111]

                    #ecu 11.2
                    indexCond112 = np.intersect1d(indexCond11, np.argwhere(Fz11 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.2
                    poblacion[indexCond112] = z11[indexCond112]

        
        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin,fitness,solutionsRanking = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS)


        #Conservo el Best
        if fitness[bestRowAux] > BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixBin[bestRowAux] = BestBinary

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)
        BestFitnes = str(np.min(fitness))

        walltimeEnd = np.round(time.time() - timerStart,6)
        processTimeEnd = np.round(time.process_time()-processTime,6) 

        dataIter = {
            "id_ejecucion": id,
            "numero_iteracion":iter,
            "fitness_mejor": BestFitnes,
            "parametros_iteracion": json.dumps({
                "fitness": BestFitnes,
                "clockTime": walltimeEnd,
                "processTime": processTimeEnd,
                "DS":DS,
                "Diversidades":  str(diversidades),
                "PorcentajeExplor": str(PorcentajeExplor),
                #"PorcentajeExplot": str(PorcentajeExplot),
                #"state": str(state)
                })
                }

        memory.append(dataIter)
       

        if iter % 100 == 0:
            memory = connect.insertMemory(memory)

    # Si es que queda algo en memoria para insertar
    if(len(memory)>0):
        memory = connect.insertMemory(memory)

    #Actualizamos la tabla resultado_ejecucion, sin mejor_solucion
    memory2 = []
    dataResult = {
        "id_ejecucion": id,
        "fitness": BestFitnes
        #"inicio": timerStartResult,
        #"fin": time.time()
        }
    memory2.append(dataResult)
    dataResult = connect.insertMemoryBest(memory2)

    # Update ejecucion
    if not connect.endEjecucion(id,datetime.now(),'terminado'):
        return False

    return True