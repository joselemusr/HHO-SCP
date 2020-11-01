__author__ = 'INVESTIGACION'

import numpy as np

def importaMatrix(fo, paramLectura):
    i = 0
    k = 0
    fila = 0
    estado = 0
    suma = 0
    for line in fo.readlines():
        linea = line.split(' ')
        linea.remove('')
        linea.remove('\n')
        if i == 0:
            row = int(linea[0])
            col = int(linea[1])
            pesos = np.zeros(col)
            matrix = np.zeros((row, col))

        #elif i > 0 and i <85:
        #elif i > 0 and i <668:
        elif i > 0 and i <paramLectura:

            for j in range(0,len(linea)):
                pesos[k] = linea[j]
                k = k + 1
        else:
            if estado == 0:
                cuenta = int(linea[0])
                estado = 1
            else:
                suma = suma + len(linea)

                for h in range(0,len(linea)):
                    matrix[fila,int(linea[h])-1] = 1
                if suma == cuenta:
                    estado = 0
                    suma = 0
                    fila = fila + 1

        i = i + 1
    return pesos, matrix

def importaMatrixRed(fo, paramLectura):
    i = 0
    k = 0
    fila = 0
    estado = 0
    suma = 0
    for line in fo.readlines():
        linea = line.replace('\n','').split(' ')
        linea.remove('')
        if i == 0:
            row = int(linea[0])
            col = int(linea[1])
            pesos = np.zeros(col)
            matrix = np.zeros((row, col))

        #elif i > 0 and i <85:
        #elif i > 0 and i <668:
        elif i > 0 and i <paramLectura:

            for j in range(0,len(linea)):
                pesos[k] = linea[j]
                k = k + 1
        else:
            if estado == 0:
                cuenta = int(linea[0])
                estado = 1
            else:
                suma = suma + len(linea)

                for h in range(0,len(linea)):
                    #print('linea h',linea[h])
                    matrix[fila,int(linea[h])-1] = 1
                if suma == cuenta:
                    estado = 0
                    suma = 0
                    fila = fila + 1

        i = i + 1
    return pesos, matrix


def generaMatrix(file):
    """
    Encargada de leer un archivo donde pasamos el nombre (file) y el directorio (dirIn)
    """
    # "paramLectura" depende de cada clase de problemas(ej:H) no reducidas y de cada instancia(ej:H.1) en las reducidas.
    paramLectura = 335 # Para el caso de los problemas E,F
    #paramLectura = 668 # Para el caso de los problemas G,H
    #paramLectura = 85 # Para el caso de los problemas pequenos
    #Casos reducido
    #paramLectura = 227 #H5
    #paramLectura = 234 #H1
    fo = open(file ,'r')
    pesos, matrix = importaMatrix(fo, paramLectura)
    #pesos, matrix = importaMatrixRed(fo, paramLectura)
    return pesos, matrix

# Forma de Uso de esta libreria
#file = 'scpnrg3.txt'
#dirIn= 'C:/Optimization/SCP/OR/G/'
#pesos, matrix = generaMatrix(file,dirIn)
#print pesos, len(pesos)
#print matrix, matrix.shape
