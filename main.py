
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


### Buscamos experimentos pendientes
connect = Database.Database()


# Algorithms

from Metaheuristics.GWO_SCP import GWO_SCP
from Metaheuristics.GWOQL_SCP import GWOQL_SCP
from Metaheuristics.SCA_SCP import SineCosine_SCP
from Metaheuristics.SCAQL_SCP import SineCosineQL_SCP


flag = True
while flag:

    id, algorithm, params = connect.getLastPendingAlgorithm('pendiente')
    
    if id == 0:
        print('No hay más ejecuciones pendientes')
        break
        
       
    print("------------------------------------------------------------------------------------------------------------------\n")
    print(f'Id Execution: {id} -  {algorithm}')
    print(json.dumps(params,indent=4))
    print("------------------------------------------------------------------------------------------------------------------\n")

    if(algorithm == 'GWO_SCP'):
        if  GWO_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme']
                )== True:
            print(f'Ejecución {id} completada ')

    if(algorithm == 'GWOQL_SCP'):
        if  GWOQL_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                0.1,
                0.4
                ) == True:
            print(f'Ejecución {id} completada ')

    if(algorithm == 'SCA_SCP'):
        if  SineCosine_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme']
                ) == True:
            print(f'Ejecución {id} completada ')

    if(algorithm == 'SCAQL_SCP'):
        if  SineCosineQL_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                0.1,
                0.4
                ) == True:
            print(f'Ejecución {id} completada ')

  