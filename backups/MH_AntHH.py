
#%%
from Problem.util import read_instance as Instance
from Problem import SCP
from pathlib import Path
from os import path as pathcheck

import numpy as np

class AntHH():
    '''
    ************************+*********   Parámetros *************************************
    # alfa: control relativo de rastro de feromona
    # beta: control relativo de visibilidad
    # p: factor de evaporación feromona.
    # ti: feromona item i hormiga k
    # delta Tik: { G(Lk): si item i es agregado, 0 en otro caso } .
    # G(Lk): En Minimización: Q/Lk, En Maximización Q*Lk, donde Lk = fitness. Q = parámetro
    # Q: 1/sum.i(valor aportado por item i), parametro para  los item.
    # iteraciones: máximo de iteraciones.
    # ants: cantidad de hormigas
    '''

    def __init__(self,instancia,ants,iterMax):

        self.ants = ants
        self.iterMax = iterMax
        self.maxSteps = 1
        self.evaporacion = 0.3
        self.alfa = 1
        self.beta = 5

        #instancia
        rootdir = str(Path().cwd()) + '/'
        pathInstance = rootdir + 'Problem/Instances/SCP/'

        if not pathcheck.exists(pathInstance + instancia):
            print("No se encontró la instancia: " + pathInstance + instancia)
            return False

        instance = Instance.Read(pathInstance + instancia)

        self.matrizCobertura = np.array(instance.get_r())
        self.vectorCostos = np.array(instance.get_c())

        #acciones a escoger

        transferFunction = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
        operatorBinarization = ['Standard']

        self.Metric_k = [np.inf]*self.ants
        self.DS_actions = [tf + "," + ob for tf in transferFunction for ob in operatorBinarization]

        self.Prob_tkx = np.ones(shape=(self.iterMax, self.ants,len(self.DS_actions)), dtype=np.float64)
        self.Phe_x = np.ones(shape=(len(self.DS_actions)), dtype=np.float64)

        #Variable decision hormiga
        self.X_tk = np.zeros(shape=(self.iterMax,self.ants, len(self.DS_actions)), dtype=np.int)

        self.bestMetric = np.inf

    def HH(self):

        for iter in range(0,self.iterMax):

            print("----------- iter "+str(iter)+" -----------------------")

            for ant in range(0,self.ants):

                print("----------- ant " + str(iter) + " -----------------------")

                step = 0

                while step < self.maxSteps:

                    #Update pheromona
                    self.Pheromona(iter)

                    #Seleccion esquema segun Probs
                    esquema = self.seleccionRuleta(iter,ant)


                    solucion_bin, fitness = SCP.SCP(solucion_random, self.vectorCostos, self.matrizCobertura, self.DS_actions[esquema])

                    if self.Metric_k[ant] > fitness:
                        self.Metric_k[ant] = fitness
                        self.X_tk[iter][ant][esquema] = 1


                    if fitness < self.bestMetric:
                        self.bestMetric = fitness

                    print(self.bestMetric)
                    #print(self.G(iter,ant))

                    #print(self.DS_actions[esquema])

                    step += 1

    #por definir o dejar en n=1
    def infoHeuristica(self):

        return 1


    def seleccionRuleta(self,iter,ant):


        random = np.random.uniform(0,2)

        probs = self.Probabilidad(iter,ant)
        #self.Prob_tkx[iter][ant] = probs.copy()
        print(probs)

        selected = 0
        for i, prob in enumerate(probs):
            random -= prob
            if random <= 0:
                selected = i
                break

        return selected


    def Probabilidad(self,iter,ant):

        heuristica = self.infoHeuristica()

        # DENOMINADOR
        prob_sum = 0
        for ds in range(len(self.X_tk[iter][ant])):  # [0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 ]
            prob_sum +=  (self.Phe_x[ds] ** self.alfa) * (heuristica ** self.beta)

        for ds in range(len(self.X_tk[iter][ant])):
            self.Prob_tkx[iter][ant][ds] = np.divide(((self.Phe_x[ds] ** self.alfa) * (heuristica ** self.beta)), prob_sum)

        return self.Prob_tkx[iter][ant]

    def Pheromona(self,t):

        dTx = 0
        for ant in range(0, self.ants):
            dTx += self.X_tk[t][ant] * self.G(ant)

        self.Phe_x = (1 - self.evaporacion) * self.Phe_x[t] + dTx

        return True


    def G(self,ant):


        fitness = self.Metric_k[ant]
        #Q = np.sum(self.matrizCobertura*self.X_tk[iter][ant])

        # Contexto: Minimizacion
        G = 1 / fitness

        return G




#%%
ACO = AntHH('mscp/mscp41.txt',20,200)

ACO.HH()