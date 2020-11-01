
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

algorithms = ['HHO_SCP','HHOQL_SCP']
#instances = ['mscp41']
instances = ['mscp41','mscp42','mscp43','mscp44','mscp45','mscp46','mscp47','mscp48','mscp49','mscp410','mscp51','mscp52','mscp53','mscp54','mscp55','mscp56','mscp57','mscp58','mscp59','mscp510','mscp61','mscp62','mscp63','mscp64','mscp65','mscpa1','mscpa2','mscpa3','mscpa4','mscpa5','mscpb1','mscpb2','mscpb3','mscpb4','mscpb5','mscpc1','mscpc2','mscpc3','mscpc4','mscpc5','mscpd1','mscpd2','mscpd3','mscpd4','mscpd5',]
runs = 10
maxIter = 5000
#maxIter = 200
ql_alpha = 0.1
ql_gamma =  0.4
population  = 20
repair = 3 # 1:Simple; 2:Compleja; 3:RepairGPU
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
                    'ql_alpha': ql_gamma,
                    'ql_gamma': ql_gamma,
                    'repair': repair
            }),
                'estado' : 'pendiente'
            }
            result = connection.execute(insertDatosEjecucion,data)
            idEjecucion = result.fetchone()[0]
            print(f'Poblado ID #:{idEjecucion}')

print("Todo poblado")