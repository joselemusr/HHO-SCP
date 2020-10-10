# Utils

import sys
import os
import settings
from envs import env
import numpy as np
import time
from datetime import datetime
from pathlib import Path

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

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+env('DIR_INSTANCES')

connect = Database.Database()


transferFunction = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
operatorBinarization = ['Standard','Complement','Elitist','Static','ElitistRoulette']

DS_actions = [tf + "," + ob for tf in transferFunction for ob in operatorBinarization]

def SineCosineQL_SCP(id,instance_file,instance_dir,population,maxIter,discretizacionScheme,ql_alpha,ql_gamma):

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
    poblacion = np.random.uniform(low=0.0, high=1.0, size=(pob,dim))
    matrixBin = np.zeros((pob,dim))
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)

    # QLEARNING 
    agente = QLearning.QAgent(ql_alpha, ql_gamma, DS_actions, maxIter+1)
    DS = agente.getAccion(0)

    matrixBin,fitness,solutionsRanking  = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS_actions[DS])
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)

    timerStart = time.time()
    memory = []
    for iter in range(0, maxIter):
        processTime = time.process_time()  

        if iter == 0:
            if not connect.startEjecucion(id,datetime.now(),'ejecutando'):
                return False
     
        timerStart = time.time()
        

        
        r1 = a - iter * (a/maxIter)
        r4 = np.random.uniform(low=0.0,high=1.0, size=poblacion.shape[0])
        r2 = (2*np.pi) * np.random.uniform(low=0.0,high=1.0, size=poblacion.shape)
        r3 = np.random.uniform(low=0.0,high=2.0, size=poblacion.shape)
        bestRowAux = solutionsRanking[0]
        Best = poblacion[bestRowAux]
        BestBinary = matrixBin[bestRowAux]
        BestFitness = np.min(fitness)
        poblacion[r4<0.5] = poblacion[r4<0.5] + np.multiply(r1,np.multiply(np.sin(r2[r4<0.5]),np.abs(np.multiply(r3[r4<0.5],Best)-poblacion[r4<0.5])))
        poblacion[r4>=0.5] = poblacion[r4>=0.5] + np.multiply(r1,np.multiply(np.cos(r2[r4>=0.5]),np.abs(np.multiply(r3[r4>=0.5],Best)-poblacion[r4>=0.5])))
        # poblacion[bestRow] = Best
        

        # Escogemos esquema desde QL
        DS = agente.getAccion(iter)

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin,fitness,solutionsRanking = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS_actions[DS])

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
                "PorcentajeExplor": str(PorcentajeExplor),
                "PorcentajeExplot": str(PorcentajeExplot),
                "state": str(state)
                })
                }

        memory.append(dataIter)
       
        if iter % 100 == 0:
            print(dataIter)

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