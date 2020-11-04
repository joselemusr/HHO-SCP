#  Author: Diego Tapia R.
#  E-mail: root.chile@gmail.com - diego.tapia.r@mail.pucv.cl
import numpy as np

class Q_Learning():

    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, gamma, actions, stateQ, qlAlphaType, rewardType, iterMax,epsilon = 0.05, qlAlpha =0.1):

        self.gamma = gamma
        self.qlAlphaType = qlAlphaType
        self.rewardType = rewardType
        self.iterMax = iterMax
        
        self.epsilon = epsilon
        self.qlAlpha = qlAlpha
        self.bestMetric = 999999999 #Esto en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
        self.Qvalues = np.zeros(shape=(stateQ,len(actions))) #state,actions
        self.visitas = np.zeros(shape=(stateQ,len(actions))) #state,actions
        #self.Qvalues[0] = np.zeros(len(self.actions))

    def getReward(self,metric):
        
        if self.rewardType == "withPenalty1": 
            if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                self.bestMetric = metric
                return 1
            return -1

        elif self.rewardType == "withoutPenalty1":
            if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                self.bestMetric = metric
                return 1
            return 0

    def getAccion(self,state,policy):
        
        # e-greedy
        if policy == "e-greedy":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad <= self.epsilon: #seleccion aleatorio
                return np.random.randint(low=0, high=self.Qvalues.shape[0]) #seleccion aleatoria de una accion     
            else: #selecion de Q_Value mayor        
                maximo = np.amax(self.Qvalues,axis=1) # retorna el elemento mayor por fila        
                indices = np.where(self.Qvalues[state,:] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado        
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 
        
        # greedy
        elif policy == "greedy":
            return np.argmax(self.Qvalues[state])

        # e-soft 
        elif policy == "e-soft":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad > self.epsilon: #seleccion aleatorio
                return np.random.randint(low=0, high=self.Qvalues.shape[0]) #seleccion aleatoria de una accion     
            else: #selecion de Q_Value mayor        
                maximo = np.amax(self.Qvalues,axis=1) # retorna el elemento mayor por fila        
                indices = np.where(self.Qvalues[state,:] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado        
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 

        # softMax seleccion ruleta
        elif policy == "softMax-rulette":
            #*** Falta generar una normalización de las probabilidades que sumen 1, para realizar el choice
            Qtable_normalizada = np.nan_to_num(self.Qvalues[state] / np.linalg.norm(self.Qvalues[state])) # normalizacion de valores   
            seleccionado = np.random.choice(self.Qvalues[state],p=Qtable_normalizada)
            indices = np.where(self.Qvalues[state,:] == seleccionado)[0]
            return np.random.choice(indices)
    
        # softmax seleccion ruleta elitista (25% mejores acciones)
        elif policy == "softMax-rulette-elitist":
            sort = np.argsort(self.Qvalues[state]) # argumentos ordenados
            cant_mejores = int(sort.shape[0]*0.25) # obtenemos el 25% de los mejores argumentos
            rulette_elitist = sort[0:cant_mejores] # tiene el 25% de los mejores argumentos
            return np.random.choice(rulette_elitist)
    


    def actualizar_Visitas(self,action,state): # ACTUALIZACION DE LAS VISITAS
        self.visitas[state,action] = self.visitas[state,action] + 1


    def getAlpha(self,state,action,iter):

        if self.qlAlphaType == "static": 
            #alpha estatico 
            return  self.qlAlpha

        elif self.qlAlphaType == "iteration":
            return 1 - (0.9*(iter/self.iterMax))
            
        elif self.qlAlphaType == "visits":
            return (1/(1 + self.visitas[state,action]))

    def updateQtable(self,metric,action,state,oldState,iter):

        #revisar

        Reward = self.getReward(metric)
        
        alpha = self.getAlpha(oldState,action,iter)

        Qnuevo = ( (1 - alpha) * self.Qvalues[oldState][action]) + alpha * (Reward + (self.gamma  * max(self.Qvalues[state])))

        self.actualizar_Visitas(action,oldState) #Actuzación de visitas

        #REVISAR ESTO
        self.Qvalues[state][action] = Qnuevo

    def getQtable(self):
        return self.Qvalues