import numpy as np

#Gracias Mauricio Y Lemus!
from .repair import ReparaStrategy as repara
from Discretization import DiscretizationScheme as DS

#action : esquema de discretizacion DS
def SCP(poblacion,matrixBin,solutionsRanking,costos,cobertura,ds):

    #Binarización de 2 pasos

    ds = ds.split(",")
    ds = DS.DiscretizationScheme(poblacion,matrixBin,solutionsRanking,ds[0],ds[1])
    matrixBin = ds.binariza()

    #Reparamos
    repair = repara.ReparaStrategy(cobertura,costos,cobertura.shape[0],cobertura.shape[1])
    for solucion in range(matrixBin.shape[0]):
        if repair.cumple(matrixBin[solucion]) == 0:
            matrixBin[solucion] = repair.repara_one(matrixBin[solucion])[0]

    #Calculamos Fitness
    fitness = np.sum(np.multiply(matrixBin,costos),axis=1)
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness


    return matrixBin,fitness,solutionsRanking
