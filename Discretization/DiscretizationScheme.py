import time
import numpy as np
import math
from scipy import special as scyesp

"""
Realiza una transformación de una matrix en el dominio de los continuos a una matriz binaria.
Usada para discretización de soluciones en metaheurísticas. 
Donde cada fila es una solución (individuo de la población), y las columnas de la matriz corresponden a las dimensiones de la solución.
Params
----------
matrixCont : matriz de continuos.
matrixBin  : matriz de poblacion binarizada en t-1. 
bestRowBin : individuo (fila) de mejor fitness.
SolutionRanking: Lista de indices ordenadas por fitness, en la posición 0 esta el best
transferFunction: Funciones de transferencia (V1,..,V4, S1,..,S4)
binarizationOperator: Operador de binarización (Standard,Complement, Elitist, Static, Roulette)
Returns
-------
matrixBinOut: matriz binaria.

Definiciones
---------
Para t=0, es decir, si matrixBin es vacía, se utiliza por defecto binarizationOperator = Standard.
"""

class DiscretizationScheme:
    
    def __init__(self, matrixCont, matrixBin, SolutionRanking, transferFunction, binarizationOperator):

        self.transferFunction = transferFunction
        self.binarizationOperator = binarizationOperator

        #self.matrixCont = np.ndarray(matrixCont.shape, dtype=float, buffer=matrixCont)
        self.matrixCont = matrixCont
        #self.matrixBin = np.ndarray(matrixBin.shape, dtype=float, buffer=matrixBin)
        self.matrixBin = matrixBin
        self.SolutionRanking = SolutionRanking
        self.bestRow = np.argmin(SolutionRanking) 

        #output
        self.matrixProbT = np.zeros(self.matrixCont.shape)
        self.matrixBinOut = np.zeros(self.matrixBin.shape)

        # debug
        self.matrixProbTAux = np.zeros(self.matrixCont.shape)
        self.matrixBinOutAux = np.zeros(self.matrixBin.shape)

    #Funciones de Transferencia

    def T_V1(self):
        # revisado ok
        self.matrixProbT = np.abs(scyesp.erf(np.divide(np.sqrt(np.pi),2)*self.matrixCont))

    def T_V2(self):
        self.matrixProbT = np.abs(np.tanh(self.matrixCont))

    def T_V3(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbT[i][d] = math.fabs(self.matrixCont[i][d]/math.sqrt(1+(self.matrixCont[i][d]**2)))
        
        # print(self.matrixProbT)
        
        self.matrixProbT = np.abs(np.divide(self.matrixCont, np.sqrt(1+np.power(self.matrixCont,2))))

    def T_V4(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = math.fabs((2/math.pi)*math.atan((math.pi/2)*self.matrixCont[i][d]))
        
        self.matrixProbT = np.abs( np.divide(2,np.pi)*np.arctan( np.divide(np.pi,2)*self.matrixCont ))

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S1(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(-2 * self.matrixCont[i][d]))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(-2*self.matrixCont) ) )

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S2(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(-1 * self.matrixCont[i][d]))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(-1*self.matrixCont) ) )

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S3(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(np.divide(-1*self.matrixCont[i][d],2)))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(-1*self.matrixCont,2) ) ))

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S4(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(np.divide(-1*self.matrixCont[i][d],3)))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(-1*self.matrixCont,3) ) ))

        # return (self.matrixProbTAux-self.matrixProbT)

    #Binarization
    def B_Standard(self):

        # for i in range(len(self.matrixProbT)):
            
        #     for d in range(len(self.matrixProbT[i])):
        #         #rand = np.random.uniform(0, 1)

        #         if rand < self.matrixProbT[i][d]:
        #             self.matrixBinOut[i][d] = 1
        #         else:
        #             self.matrixBinOut[i][d] = 0

        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        self.matrixBinOut = np.greater(self.matrixProbT,matrixRand).astype(int)

    def B_Complement(self):

        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        matrixComplement = np.abs(1-self.matrixBin)
        # cambie greater, por greater_equal (por ecuacion)
        self.matrixBinOut = np.multiply(np.greater_equal(self.matrixProbT,matrixRand).astype(int),matrixComplement)
        return self.matrixBinOut

    def B_Elitist(self):

        # start = time.time()

        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        # greater, porque es estricto en la ecuacion.
        conditionMatrix = np.greater(self.matrixProbT,matrixRand)
        #todo: validar que el index exista
        bestIndividual = self.matrixBin[self.bestRow]
        # si ProbT > Rand() , then bestIndividualBin, else 0
        self.matrixBinOut = np.where(conditionMatrix==True,bestIndividual,0)

        # print("-----------------------------")

        # print(bestIndividual)

        # print("-----------------------------")

        # print("Test Numpy:")
        # print(self.matrixBinOut)

        # print(time.time()-start)

        # start = time.time()
        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):

        #         if matrixRand[i][d] < self.matrixProbT[i][d]:
        #             self.matrixBinOutAux[i][d] = bestIndividual[d]
        #         else:
        #             self.matrixBinOutAux[i][d] = 0

        # print("Test For:")
        # print(self.matrixBinOutAux)
        # print(time.time()-start)

        # print ("igualdad a cero:")
        # dif = (self.matrixBinOut-self.matrixBinOutAux)
        # print((dif==0).all())

    def B_Static(self):
        
        # start = time.time()
        alfa = 1/3
        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):

        #         if (self.matrixProbT[i][d] > alfa) and (self.matrixProbT[i][d] <= 0.5*(1+alfa)):
        #              self.matrixBinOutAux[i][d] = self.matrixBin[i][d]
        #         elif self.matrixProbT[i][d] <= alfa:
        #             self.matrixBinOutAux[i][d] = 0
        #         else:
        #             self.matrixBinOutAux[i][d] = 1

        # print("Test For:")
        # print(self.matrixBinOutAux)
        # print(time.time()-start)

        # start = time.time()
        
        self.matrixBinOut[self.matrixProbT<=alfa] = 0
        self.matrixBinOut[(self.matrixProbT > alfa) & (self.matrixProbT <= 0.5*(1+alfa))] = self.matrixBin[(self.matrixProbT > alfa) & (self.matrixProbT <= 0.5*(1+alfa))]
        self.matrixBinOut[self.matrixProbT>=0.5*(1+alfa)] = 1

        # print("Test Asignacion directa:")
        # print(self.matrixBinOut)
        # print(time.time()-start)

        # print ("igualdad a cero:")
        # dif = (self.matrixBinOut-self.matrixBinOutAux)
        # print((dif==0).all())

        # def ElitistRoulette(self):

        #     for i in range(len(self.solution)):

        #         rand = np.random.uniform(0,1)

        #         if rand < self.probT[i]:
        #             self.solution_int[i] = 1
        #         else:
        #             self.solution_int[i] = 0

        #     return self.solution_int

    def B_ElitistRoulette(self):
        
        start = time.time()

        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        #greater, porque es estricto en la ecuacion.
        conditionMatrix = np.greater(self.matrixProbT,matrixRand)
        #todo: validar que el index exista

        alfa = 0.2
        # condicion sum()==0, para el caso en que entregamos lista de rank con [0 0 0 0 0 0 0 0 ... 0 0 0 0] -> cuando generamos poblacion inicial
        if (self.SolutionRanking.shape[0]*alfa < 1) or (self.SolutionRanking.sum()==0):
            #bestIndividual = self.matrixBin[self.SolutionRanking[0]]
            bestIndividual = self.matrixBin[0]
        else:
            BestSolutionRaking = int(self.SolutionRanking.shape[0] * alfa)
            random = np.random.randint(low = 0, high = BestSolutionRaking)
            bestIndividual = self.matrixBin[random]

        # si ProbT > Rand() , then bestIndividualBin, else 0
        self.matrixBinOut = np.where(conditionMatrix==True,bestIndividual,0)

        # print(self.matrixBinOut)

        # print("-----------------------------")

        # print(bestIndividual)

        # print("-----------------------------")

        # print("Test Numpy:")
        # print(self.matrixBinOut)

        # print(time.time()-start)

        # start = time.time()
        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):

        #         if matrixRand[i][d] < self.matrixProbT[i][d]:
        #             self.matrixBinOutAux[i][d] = bestIndividual[d]
        #         else:
        #             self.matrixBinOutAux[i][d] = 0

        # print("Test For:")
        # print(self.matrixBinOutAux)
        # print(time.time()-start)

        # print ("igualdad a cero:")
        # dif = (self.matrixBinOut-self.matrixBinOutAux)
        # print((dif==0).all())

    def binariza(self):

        if self.transferFunction == 'V1':
            self.T_V1()

        if self.transferFunction == 'V2':
            self.T_V2()

        if self.transferFunction == 'V3':
            self.T_V3()

        if self.transferFunction == 'V4':
            self.T_V4()

        if self.transferFunction == 'S1':
            self.T_S1()

        if self.transferFunction == 'S2':
            self.T_S2()

        if self.transferFunction == 'S3':
            self.T_S3()

        if self.transferFunction == 'S4':
            self.T_S4()

        if self.binarizationOperator == 'Standard':
            self.B_Standard()

        if self.binarizationOperator == 'Complement':
            self.B_Complement()

        if self.binarizationOperator == 'Elitist':
            self.B_Elitist()

        if self.binarizationOperator == 'Static':
            self.B_Static()

        if self.binarizationOperator == 'ElitistRoulette':
            self.B_ElitistRoulette()

        return self.matrixBinOut
