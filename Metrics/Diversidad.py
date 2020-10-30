import numpy as np
import math

#action : esquema de discretizacion DS
def MomentoDeInercia(Poblacion):
  Diversidad = 0
  N = Poblacion.shape[0]
  D = Poblacion.shape[1]
  promedio = np.mean(Poblacion, axis=0)
  MatrizDiversidad = np.power((Poblacion - promedio),2)

  Diversidad = np.sum(MatrizDiversidad)

  return Diversidad

def Hamming(Poblacion):
  Diversidad = 0
  frecuencias0 = []
  frecuencias1 = []
  
  for d in range(len(Poblacion[0])):
    frecuencia0 = 0
    frecuencia1 = 0
    
    for p in range(len(Poblacion)):
      if Poblacion[p][d] == 0:
        frecuencia0 = frecuencia0 + 1
      else:
        frecuencia1 = frecuencia1 + 1
    
    frecuencias0.append(frecuencia0)
    frecuencias1.append(frecuencia1)

  sumatoria = 0
  for d in range(len(Poblacion[0])):
    n = len(Poblacion)
    sumatoria = sumatoria + (frecuencias0[d]/n) * (1 - (frecuencias0[d]/n))
    sumatoria = sumatoria + (frecuencias1[d]/n) * (1 - (frecuencias1[d]/n))

  Diversidad = ((len(Poblacion)**2) / (2 * len(Poblacion[0]))) * sumatoria

  return Diversidad

def Entropica(Poblacion):
  Diversidad = 0
  frecuencias0 = []
  frecuencias1 = []
  
  for d in range(len(Poblacion[0])):
    frecuencia0 = 0
    frecuencia1 = 0
    
    for p in range(len(Poblacion)):
      if Poblacion[p][d] == 0:
        frecuencia0 = frecuencia0 + 1
      else:
        frecuencia1 = frecuencia1 + 1
    
    frecuencias0.append(frecuencia0)
    frecuencias1.append(frecuencia1)

  sumatoria = 0
  for d in range(len(Poblacion[0])):
    n = len(Poblacion)
    if frecuencias0[d] != 0 and frecuencias1[d] != 0:
      sumatoria = sumatoria + (frecuencias0[d]/n) * (math.log(frecuencias0[d]/n))
      sumatoria = sumatoria + (frecuencias1[d]/n) * (math.log(frecuencias1[d]/n))

  Diversidad = (-1 / (len(Poblacion[0]))) * sumatoria

  return Diversidad


def LeungGaoXu(Poblacion):
  Diversidad = 0
  frecuencias0 = []
  frecuencias1 = []
  n = len(Poblacion)
  for d in range(len(Poblacion[0])):
    frecuencia0 = 0
    frecuencia1 = 0
    
    for p in range(len(Poblacion)):
      if Poblacion[p][d] == 0:
        frecuencia0 = frecuencia0 + 1
      else:
        frecuencia1 = frecuencia1 + 1
    
    frecuencias0.append(frecuencia0/n)
    frecuencias1.append(frecuencia1/n)

  sumatoria = 0
  for d in range(len(Poblacion[0])):
    
    sumatoria = sumatoria + g(frecuencias0[d]) * g(1- frecuencias0[d])

  Diversidad =  sumatoria

  return Diversidad

def g(frecuencia):

  if frecuencia == 0 or frecuencia == 1:
    g = frecuencia
  else:
    g = 1

  return g

def Dimensional(Poblacion):
  Diversidad = 0
  MatrizDiversidad = np.zeros((len(Poblacion)))
  Pob = np.array(Poblacion)
  Promedio = np.median(Poblacion, axis=1)

  for d in range(len(Poblacion[0])):
    Divj = 0
    MatrizDiversidad = abs(Promedio  - Pob[:,d])
    Diversidad = Diversidad + MatrizDiversidad.sum()/len(Poblacion[0])
        
  Diversidad = Diversidad / len(Poblacion)

  return Diversidad

def PesosDeInercia(Poblacion):
  Pob = Poblacion
  N = Pob.shape[0]
  D = Pob.shape[1]
  promedio = np.mean(Poblacion, axis=0)
  
  MatrizDiversidad = np.divide((np.sqrt(np.sum((np.power((Pob - promedio),2)), axis=1))),N)
  Diversidad = np.sum(MatrizDiversidad)

  return Diversidad

def DimensionalHussain(Poblacion):
  Pob = np.array(Poblacion)
  N = Pob.shape[0]
  D = Pob.shape[1]
  Medias = np.mean(Poblacion, axis=0)
  
  MatrizDiversidad = np.divide(np.divide(np.abs(Medias - Pob),N),D)
    
  Diversidad = np.sum(MatrizDiversidad)

  return Diversidad

def ObtenerDiversidadYEstado(Poblacion,maxDiversidades):
    #Calculamos las diversidades
    diversidades = []
    diversidades.append(DimensionalHussain(Poblacion)) #0
    diversidades.append(PesosDeInercia(Poblacion)) #1
    diversidades.append(Dimensional(Poblacion)) #2
    diversidades.append(LeungGaoXu(Poblacion)) #3
    diversidades.append(Entropica(Poblacion)) #4
    diversidades.append(Hamming(Poblacion)) #5
    diversidades.append(MomentoDeInercia(Poblacion)) #6     

    #Actualizar maxDiversidades y calculamos PorcentajeExplor PorcentajeExplot
    PorcentajeExplor = []
    PorcentajeExplot = []
    state = []
    for i in range(len(diversidades)):
        if diversidades[i] > maxDiversidades[i]:
            maxDiversidades[i] = diversidades[i]
        #calculamos PorcentajeExplor PorcentajeExplot 
        PorcentajeExplor.append((diversidades[i]/maxDiversidades[i])*100)
        PorcentajeExplot.append((abs(diversidades[i]-maxDiversidades[i])/maxDiversidades[i])*100)

        #Determinar estado
        if PorcentajeExplor[i] >= PorcentajeExplot[i]:
            state.append(1) # Exploración
        else:
            state.append(0) # Explotación

    #return diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state
    return np.around(diversidades,2), np.around(maxDiversidades,2), np.around(PorcentajeExplor,2), np.around(PorcentajeExplot,2), state