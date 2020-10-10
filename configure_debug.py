
# Utils

import sys
import os
import settings
from envs import env
import numpy as np
import time
from pathlib import Path

# SQL
import sqlalchemy as db
import psycopg2
import json

# Definicion Environments Vars
workdir = env('PWD')
db_motor = env('DB_MOTOR')
db_user = env('DB_USER')
db_pass = env('DB_PASS')
db_server = env('DB_SERVER')
db_port = env('DB_PORT')
db_base   = env('DB_BASE')


# Conexión a la DB de resultados
engine = db.create_engine(f'{db_motor}://{db_user}:{db_pass}@{db_server}:{db_port}/{db_base}')
metadata = db.MetaData()

try: 
    connection = engine.connect()

except db.exc.SQLAlchemyError as e:
    exit(str(e.__dict__['orig']))




datosEjecucion = db.Table('datos_ejecucion', metadata, autoload=True, autoload_with=engine)
insertDatosEjecucion = datosEjecucion.insert().returning(datosEjecucion.c.id)

algorithms = ['SCA_SCP','SCAQL_SCP','GWO_SCP','GWOQL_SCP']
instances = ['mscp41','mscp51','mscp61','mscpa1','mscpb1','mscpc1','mscpd1','mscpnre1','mscpnrf1']
runs = 10
maxIter = 10
maxIter = 100
ql_alpha = 0.1
ql_gamma =  0.4
population  = 20
instance_dir = "MSCP/"
for instance in instances:
    for algorithm in algorithms:
        for run in range(runs):
            data = {
                'nombre_algoritmo' : algorithm,

                'parametros': json.dumps({
                    'instance_name' : instance,
                    'instance_file': instance+'.txt',
                    'instance_dir': instance_dir,
                    'population': population,
                    'maxIter':maxIter,
                    'discretizationScheme':'V4,Elitist',
                    'ql_alpha':0.1,
                    'ql_gamma':0.4
            }),
                'estado' : 'pendiente'
            }
            result = connection.execute(insertDatosEjecucion,data)
            idEjecucion = result.fetchone()[0]
            print(f'Poblado ID #:{idEjecucion}')

print("Todo poblado")