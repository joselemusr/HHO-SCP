import numpy as np

#Gracias Mauricio Y Lemus!
from .repair import ReparaStrategy as repara
from Discretization import DiscretizationScheme as DS
from .util import SCPProblem
from .repair import cumpleRestricciones as cumpleGPU

#action : esquema de discretizacion DS
def SCP(poblacion,matrixBin,solutionsRanking,costos,cobertura,ds,repairType,problemaGPU,pondRestricciones):

    #Binarización de 2 pasos
    ds = ds.split(",")
    ds = DS.DiscretizationScheme(poblacion,matrixBin,solutionsRanking,ds[0],ds[1])
    matrixBin = ds.binariza()

    #Reparamos
    if repairType == 3: #Si reparamos con GPU
        matrixBin = cumpleGPU.reparaSoluciones(matrixBin,problemaGPU.instance.get_r(),problemaGPU.instance.get_c(), pondRestricciones)   

    else: #Si reparamos con CPU simple o complejo
        repair = repara.ReparaStrategy(cobertura,costos,cobertura.shape[0],cobertura.shape[1])
        for solucion in range(matrixBin.shape[0]):
            if repair.cumple(matrixBin[solucion]) == 0:
                matrixBin[solucion] = repair.repara_one(matrixBin[solucion],repairType,problemaGPU,pondRestricciones)[0]

    #Calculamos Fitness
    fitness = np.sum(np.multiply(matrixBin,costos),axis =1)
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness

    return matrixBin,fitness,solutionsRanking
