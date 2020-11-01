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

# ML
from MachineLearning import QLearning

# RepairGPU
from Problem.util import SCPProblem

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+env('DIR_INSTANCES')

connect = Database.Database()


transferFunction = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
operatorBinarization = ['Standard','Complement','Elitist','Static','ElitistRoulette']

DS_actions = [tf + "," + ob for tf in transferFunction for ob in operatorBinarization]

def HHOQL_SCP(id,instance_file,instance_dir,population,maxIter,discretizacionScheme,ql_alpha,ql_gamma,repair):

    instance_path = workdirInstance + instance_dir + instance_file

    if not os.path.exists(instance_path):
        print(f'No se encontró la instancia: {instance_path}')
        return False

    instance = Instance.Read(instance_path)

    problemaGPU = SCPProblem.SCPProblem(instance_path)
    pondRestricciones = 1/np.sum(problemaGPU.instance.get_r(), axis=1)


    matrizCobertura = np.array(instance.get_r())
    vectorCostos = np.array(instance.get_c())

    dim = len(vectorCostos)
    pob = population
    maxIter = maxIter
    DS = discretizacionScheme #[v1,Standard]

    #Variables de diversidad
    diversidades = []
    maxDiversidades = np.zeros(7) #es tamaño 7 porque calculamos 7 diversidades
    PorcentajeExplor = []
    PorcentajeExplot = []
    state = []

    #Generar población inicial
    poblacion = np.random.uniform(low=0.0, high=1.0, size=(pob,dim))
    matrixBin = np.zeros((pob,dim))
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)

    # QLEARNING 
    agente = QLearning.QAgent(ql_alpha, ql_gamma, DS_actions, maxIter+1)
    DS = agente.getAccion(0)

    matrixBin,fitness,solutionsRanking  = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS_actions[DS],repair,problemaGPU,pondRestricciones)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)


    #Parámetros fijos de HHO
    beta=1.5 #Escalar según paper
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) #Escalar
    LB = -10 #Limite inferior de los valores continuos
    UB = 10 #Limite superior de los valores continuos


    inicio = datetime.now()
    timerStartResult = time.time()
    memory = []

    for iter in range(0, maxIter):
        #print(iter)
        processTime = time.process_time()  
     
        timerStart = time.time()

        #HHOQL
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

        #print(f'Eabs: {Eabs}')

        #ecu 1.1
        indexCond1_1 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.1
        #print("indexCond1_1")
        if indexCond1_1.shape[0] != 0:
            Xrand = poblacion[np.random.randint(low=0, high=pob, size=indexCond1_1.shape[0])] #Me entrega un conjunto de soluciones rand de tam indexCond1_1.shape[0] (osea los que cumplen la cond11)
            
            poblacion[indexCond1_1] = Xrand - np.multiply(np.random.uniform(low= 0.0, high=1.0, size=indexCond1_1.shape[0]), np.abs(Xrand- (2* np.multiply(np.random.uniform(low= 0.0, high=1.0, size = indexCond1_1.shape[0]),poblacion[indexCond1_1].T).T)).T).T #Aplico la ecu 1.1 solamente a las que cumplen las condiciones np.argwhere(Eabs>=1),np.argwhere(q>=0.5)
        
        #ecu 1.2
        indexCond1_2 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q<0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.2 
        #print("indexCond1_2")
        if indexCond1_2.shape[0] != 0:
            
            array_Xm = np.zeros(poblacion[indexCond1_2].shape)
            array_Xm = array_Xm + Xm

            poblacion[indexCond1_2] = ((Best - array_Xm).T - np.multiply( np.random.uniform(low= 0.0, high=1.0, size = indexCond1_2.shape[0]), (LB + np.random.uniform(low= 0.0, high=1.0, size = indexCond1_2.shape[0]) * (UB-LB)) )).T

        #ecu 4
        indexCond4 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 4
        #print("indexCond4")
        if indexCond4.shape[0] != 0:
            
            array_Xm = np.zeros(poblacion[indexCond4].shape)
            array_Xm = array_Xm + Xm

            poblacion[indexCond4] = ((array_Xm - poblacion[indexCond4]) - np.multiply( E[indexCond4], np.abs(np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond4.shape[0])), array_Xm.T ).T - poblacion[indexCond4]).T).T)

        #ecu 10
        indexCond10 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10 
        if indexCond10.shape[0] != 0:
            #print("indexCond10")    
            y10 = poblacion
            #ecu 7
            
            Array_y10 = np.zeros(poblacion[indexCond10].shape)
            Array_y10 = Array_y10 + y10[bestRowAux]
            
            y10[indexCond10] = Array_y10- np.multiply( E[indexCond10], np.abs( np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond10.shape[0])), Array_y10.T ).T- Array_y10 ).T ).T  
            
            #ecu 8
            z10 = y10
            S = np.random.uniform(low= 0.0, high=1.0, size=(y10[indexCond10].shape))
            LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y10[indexCond10].shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y10[indexCond10].shape))),(1/beta)))
            
            z10[indexCond10] = y10[indexCond10] + np.multiply(LF,S)

            #evaluar fitness de ecu 7 y 8
            Fy10 = solutionsRanking
            Fy10[indexCond10] = Problem.SCP(y10[indexCond10],matrixBin[indexCond10],solutionsRanking[indexCond10],vectorCostos,matrizCobertura,DS_actions[DS],repair,problemaGPU,pondRestricciones)[1]
            
            Fz10 = solutionsRanking
            Fz10[indexCond10] = Problem.SCP(z10[indexCond10],matrixBin[indexCond10],solutionsRanking[indexCond10],vectorCostos,matrizCobertura,DS_actions[DS],repair,problemaGPU,pondRestricciones)[1]
            
            #ecu 10.1
            indexCond101 = np.intersect1d(indexCond10, np.argwhere(Fy10 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.1
            if indexCond101.shape[0] != 0:
                #print("indexCond101")
                poblacion[indexCond101] = y10[indexCond101]

            #ecu 10.2
            indexCond102 = np.intersect1d(indexCond10, np.argwhere(Fz10 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.2
            if indexCond102.shape[0] != 0:
                #print("indexCond102")    
                poblacion[indexCond102] = z10[indexCond102]

            # ecu 6
            indexCond6 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 6
            if indexCond6.shape[0] != 0:
                #print("indexCond6")    
                poblacion[indexCond6] = Best - np.multiply(E[indexCond6], np.abs(Best - poblacion[indexCond6] ).T ).T

            #ecu 11
            indexCond11 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11
            if indexCond11.shape[0] != 0:
                #print("indexCond11")
                #ecu 12
                y11 = poblacion
                array_Xm = np.zeros(poblacion[indexCond11].shape)
                array_Xm = array_Xm + Xm

                y11[indexCond11] = y11[bestRowAux]-  np.multiply(E[indexCond11],  np.abs(  np.multiply(  2*(1-np.random.uniform(low= 0.0, high=1.0, size=poblacion[indexCond11].shape)),  y11[bestRowAux]  )- array_Xm ).T ).T

                #ecu 13
                z11 = y11
                S = np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))
                LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y11.shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))),(1/beta)))
                z11[indexCond11] = y11[indexCond11] + np.multiply(S[indexCond11],LF[[indexCond11]])

                #evaluar fitness de ecu 12 y 13
                if solutionsRanking is None: solutionsRanking = np.ones(pob)*999999
                
                Fy11 = solutionsRanking
                Fy11[indexCond11] = Problem.SCP(y11[indexCond11],matrixBin[indexCond11],solutionsRanking[indexCond11],vectorCostos,matrizCobertura,DS_actions[DS],repair,problemaGPU,pondRestricciones)[1]

                Fz11 = solutionsRanking
                Fz11[indexCond11] = Problem.SCP(z11[indexCond11],matrixBin[indexCond11],solutionsRanking[indexCond11],vectorCostos,matrizCobertura,DS_actions[DS],repair,problemaGPU,pondRestricciones)[1]
                
                #ecu 11.1
                indexCond111 = np.intersect1d(indexCond11, np.argwhere(Fy11 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.1
                if indexCond111.shape[0] != 0:
                    poblacion[indexCond111] = y11[indexCond111]

                #ecu 11.2
                indexCond112 = np.intersect1d(indexCond11, np.argwhere(Fz11 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.2
                if indexCond112.shape[0] != 0:
                    poblacion[indexCond112] = z11[indexCond112]        

        # Escogemos esquema desde QL
        DS = agente.getAccion(iter)

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin,fitness,solutionsRanking = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS_actions[DS],repair,problemaGPU,pondRestricciones)

        #Conservo el Best
        if fitness[bestRowAux] > BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixBin[bestRowAux] = BestBinary

      
        # Observamos, y recompensa/castigo.  Actualizamos Tabla Q
        agente.Qnuevo(np.min(fitness), DS, iter+1)

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)
        BestFitnes = str(np.min(fitness)) # para JSON

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
                "DS":str(DS),
                "Diversidades":  str(diversidades),
                "PorcentajeExplor": str(PorcentajeExplor)
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
    fin = datetime.now()
    dataResult = {
        "id_ejecucion": id,
        "fitness": BestFitnes,
        "inicio": inicio,
        "fin": fin
        }
    memory2.append(dataResult)
    dataResult = connect.insertMemoryBest(memory2)

    # Update ejecucion
    if not connect.endEjecucion(id,datetime.now(),'terminado'):
        return False

    return True