#  Author: Diego Tapia R.
#  E-mail: root.chile@gmail.com - diego.tapia.r@mail.pucv.cl
import numpy as np

class QAgent():

    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, alpha, gamma, actions, state):

        self.gamma = gamma
        self.alpha = alpha
        self.state = state
        self.actions = actions
        
        self.epsilon = 0.05
        self.bestMetric = 0
        self.Qvalues = np.zeros(shape=(self.state,len(self.actions))) #state,actions
        # self.Qvalues[0] = np.random.random(size=len(actions))
        self.Qvalues[0] = np.zeros(len(self.actions))

    def getReward(self,metric):

        if self.bestMetric <= metric:

            self.bestMetric = metric

            return 1

        return -1

    def getAccion(self,state,politica):
        # if state == 0:
        #     return self.Qvalues[0][np.random.randint(0,len(self.actions)+1)] # cota sup, es exclusiva
        # politicas de selección de acciones
        
        # e-greedy
        probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
        if probabilidad <= self.epsilon: #seleccion aleatorio
            return np.random.randint(low=0, high=self.Qvalues.shape[0]) #seleccion aleatoria de una accion     
        else: #selecion de Q_Value mayor        
            maximo = np.amax(self.Qvalues,axis=1) # retorna el elemento mayor por fila        
            indices = np.where(self.Qvalues[state,:] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado        
            return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 
        
        # greedy
        return np.argmax(self.Qvalues[state])


        # e-soft 
        probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
        if probabilidad > self.epsilon: #seleccion aleatorio
            return np.random.randint(low=0, high=self.Qvalues.shape[0]) #seleccion aleatoria de una accion     
        else: #selecion de Q_Value mayor        
            maximo = np.amax(self.Qvalues,axis=1) # retorna el elemento mayor por fila        
            indices = np.where(self.Qvalues[state,:] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado        
            return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 

        # softMax seleccion ruleta
        
        Qtable_normalizada = self.Qvalues[state] / np.linalg.norm(self.Qvalues[state]) # normalizacion de valores   
        seleccionado = np.random.choice(self.Qvalues[state],p=Qtable_normalizada)
        indices = np.where(self.Qvalues[state,:] == seleccionado)[0]
        return np.random.choice(indices)
    
        # softmax seleccion ruleta elitista (25% mejores acciones)
        sort = np.argsort(self.Qvalues[state]) # argumentos ordenados
        cant_mejores = int(sort.shape[0]*0.25) # obtenemos el 25% de los mejores argumentos
        rulette_elitist = sort[0:cant_mejores] # tiene el 25% de los mejores argumentos
        return np.random.choice(rulette_elitist)
    
    
    def Qnuevo(self,metric,action,state):

        #revisar
        Qmax = max(self.Qvalues[state-1])

        R = self.getReward(metric)

        Qnuevo = ( (1 - self.alpha) * self.Qvalues[state-1][action]) + self.alpha * (R + (self.gamma  * Qmax))

        #REVISAR ESTO
        self.Qvalues[state] = self.Qvalues[state-1].copy()
        self.Qvalues[state][action] = Qnuevo

        return Qnuevo


#%%
''' FORMA DE USO 

iterMax = 10
actions = ['V1','V2','V3','V4','S1','S2','S3','S4']
actionsSelected = [np.nan]*iterMax
agente = QAgent(0.3,0.6,actions,iterMax)

print("Qvalues state "+str(0)+":"+str(agente.Qvalues[0]))
action = agente.getAccion(0)
actionsSelected[0] = action
#%%
for iter in range(1,iterMax):

    print("Qvalues state "+str(iter)+":"+str(agente.Qvalues[iter-1]))

    metric = np.random.rand(1)  # aqui el fitness obtenido
    agente.Qnuevo(metric, action, iter)

    action = agente.getAccion(iter)
    print("Action :"+str(action)+" -> "+str(actions[action]))
    actionsSelected[iter] = action


# %%

print(agente.Qvalues)

'''