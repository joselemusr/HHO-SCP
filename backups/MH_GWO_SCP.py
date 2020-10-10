# -*- coding: utf-8 -*-

#  Author: Diego Tapia R.
#  E-mail: root.chile@gmail.com - diego.tapia.r@mail.pucv.cl

########################################### EJECUCIÓN  #########################################################
# python GWO_SCP.py instancia (string)
#                   resultados(string)
#                   poblacion (int)
#                   iteraciones (int)
#                   corridas_start (int)
#                   corridas_end (int)
###############################################################################################################

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

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+env('DIR_INSTANCES')

connect = Database.Database()

def LoboGris_SCP(id,instance_file,instance_dir,population,maxIter,discretizacionScheme):

    instance_path = workdirInstance + instance_dir + instance_file

    if not os.path.exists(instance_path):
        print(f'No se encontró la instancia: {instance_path}')
        return False

    instance = Instance.Read(instance_path)

    matrizCobertura = np.array(instance.get_r())
    vectorCostos = np.array(instance.get_c())

    dim = len(vectorCostos)

    Lobos = population
    maxIter = maxIter
    DS = discretizacionScheme


    timerStart = time.time()
    

    # Alfa, beta y delta
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    # Posiciones
    posLobos = np.zeros((Lobos, dim))
    for i in range(dim):
        posLobos[:, i] = np.random.uniform(0, 1, Lobos)

    memory = []
    for iter in range(0, maxIter):
        processTime = time.process_time()  

        if iter == 0:
            if not connect.startEjecucion(id,datetime.now(),'ejecutando'):
                return False
           

        timerStart = time.time()

        for lobo in range(0, Lobos):

            # F.O por cada lobo
            posLobos[lobo, :], fitness = Problem.SCP(posLobos[lobo,:],vectorCostos,matrizCobertura,DS)

            # Actualizamos alpha,beta, delta
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = posLobos[lobo, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness
                Beta_pos = posLobos[lobo, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness
                Delta_pos = posLobos[lobo, :].copy()

        #parametro linealmente decreciente  2->0
        a = 2 - iter * ((2) / maxIter);
        # Lobos
        for lobo in range(0, Lobos):
            for j in range(0, dim):
                r1 = np.random.uniform(0,1) 
                r2 = np.random.uniform(0,1) 

                A1 = 2 * a * r1 - a; 
                C1 = 2 * r2;  

                D_alpha = abs(C1 * Alpha_pos[j] - posLobos[lobo, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = np.random.uniform(0, 1)  
                r2 = np.random.uniform(0, 1)  

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - posLobos[lobo, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = np.random.uniform(0, 1)  
                r2 = np.random.uniform(0, 1)  

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * Delta_pos[j] - posLobos[lobo, j])
                X3 = Delta_pos[j] - A3 * D_delta

                posLobos[lobo, j] = (X1 + X2 + X3) / 3

        walltimeEnd = np.round(time.time() - timerStart,6)
        processTimeEnd = np.round(time.process_time()-processTime,6) 

        dataIter = {
            "id_ejecucion": id,
            "numero_iteracion":iter,
            "fitness_mejor": Alpha_score,
            "parametros_iteracion": json.dumps({
                "fitness": Alpha_score,
                "clockTime": walltimeEnd,
                "processTime": processTimeEnd,
                "metrica1":0
            })
        }

        memory.append(dataIter)
       
        if iter % 10 == 0:
            print(dataIter)

        if iter % 100 == 0:
            memory = connect.insertMemory(memory)

    # Si es que queda algo en memoria para insertar
    if(len(memory)>0):
        memory = connect.insertMemory(memory)

    # Update ejecucion
    if not connect.endEjecucion(id,datetime.now(),'terminado'):
        return False

    return True


### Buscamos experimentos pendientes

flag = True
while flag:

    id, params = connect.getLastPending('pendiente','GWO_SCP')
    
    if id == 0:
        print('No hay más ejecuciones pendientes')
        break
        
       
    print("------------------------------------------------------------------------------------------------------------------\n")
    print(f'Id Execution: {id} ')
    print(json.dumps(params,indent=4))
    print("------------------------------------------------------------------------------------------------------------------\n")

    if LoboGris_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme']
                ) == True:

        print(f'Ejecución {id} completada ')
    else:
        print(f'Ejecución {id}: ocurrió un error al ejecutar. Se detuvo todo. ')
        flag = False