#!/usr/bin/python
# encoding=utf8
from datetime import datetime
import numpy as np
class Read():
# -*- coding: utf-8 -*-

    def __init__(self,file):
        self.__c = []
        self.__r = []
        self.rows = 0
        self.columns  = 0
        self.LeerInstancia(file)
    
    def get_c(self):
        return self.__c

    def set_c(self, c):
        self.__c = c    
        
    def get_r(self):
        return self.__r

    def set_r(self, r):
        self.__r = r    
    
    def get_rows(self):
        return self.rows
    
    def get_columns(self):
        return self.columns 
    
    def LeerInstancia(self,Instancia):
#        print(f'abriendo archivo {datetime.now()}')
        self.optimo = self.obtenerOptimo(Instancia)
        Archivo = open(Instancia, "r")
#        print(f'fin abriendo archivo {datetime.now()}')    
        # Leer Dimensión
        Registro = Archivo.readline().split()
        self.rows = int(Registro[0])
        self.columns      = int(Registro[1])
        
        # Leer Costo
        Costos        = []
        Registro      = Archivo.readline()
        ContVariables = 1
#        print(f'ciclo qlo 1 {datetime.now()}')
        while Registro != "" and ContVariables <= self.columns :
            Valores = Registro.split()
            for Contador in range(len(Valores)):
                Costos.append(int(Valores[Contador]))
                ContVariables = ContVariables + 1
            Registro = Archivo.readline()
#        print(f'fin ciclo qlo 1 {datetime.now()}')
        # Preparar Matriz de Restricciones.
        
#        print(f'ciclo qlo 2 {datetime.now()}')
        Restricciones = np.zeros((self.rows,self.columns), dtype=np.int32).tolist()
#        for Fila in range(self.rows):
#            Restricciones.append([])
#            for Columna in range(self.columns):
#                Restricciones[Fila].append(0)
#        print(f'fin ciclo qlo 2 {datetime.now()}')
        # Leer Restricciones    
        ContVariables      = 1
        Fila               = 0
        cont = 0
#        print(f'ciclo qlo 3 {datetime.now()}')
        while Registro != "":
#            if Registro != '\n': 
#            Registro = Registro.strip()
#            Registro = Registro.replace('\n','').replace(" ",',')
#            Registro = np.fromstring(Registro, dtype=int, sep=",")
#            print(np.fromstring(Registro, dtype=int, sep=","))
#            exit()
            CantidadValoresUno = int(Registro)
            ContadorValoresUno = 0
            Registro = Archivo.readline()
#            print(Registro)
            Registro = Registro.replace('\n','').replace("\\n'",'')
#            print(Registro)
            while Registro != "" and ContadorValoresUno < CantidadValoresUno: 
                Columnas = Registro.split() 
                for Contador in range(len(Columnas)):
                    Columna = int(Columnas[Contador]) - 1
                    Restricciones[Fila][Columna] = 1
                    ContadorValoresUno = ContadorValoresUno + 1
                Registro = Archivo.readline()
            Fila = Fila + 1
#        print(f'fin ciclo qlo 3 {datetime.now()}')
        Archivo.close()
        self.set_c(Costos)
        self.set_r(Restricciones)        


    def obtenerOptimo(self, archivoInstancia):
        orden = {
            'scp41':[0,429]
            ,'scp42':[1,512]
            ,'scp43':[2,516]
            ,'scp44':[3,494]
            ,'scp45':[4,512]
            ,'scp46':[5,560]
            ,'scp47':[6,430]
            ,'scp48':[7,492]
            ,'scp49':[8,641]
            ,'scp410':[9,514]
            ,'scp51':[10,253]
            ,'scp52':[11,302]
            ,'scp53':[12,226]
            ,'scp54':[13,242]
            ,'scp55':[14,211]
            ,'scp56':[15,213]
            ,'scp57':[16,293]
            ,'scp58':[17,288]
            ,'scp59':[18,279]
            ,'scp510':[19,265]
            ,'scp61':[20,138]
            ,'scp62':[21,146]
            ,'scp63':[22,145]
            ,'scp64':[23,131]
            ,'scp65':[24,161]
            ,'scpa1':[25,253]
            ,'scpa2':[26,252]
            ,'scpa3':[27,232]
            ,'scpa4':[28,234]
            ,'scpa5':[29,236]
            ,'scpb1':[30,69]
            ,'scpb2':[31,76]
            ,'scpb3':[32,80]
            ,'scpb4':[33,79]
            ,'scpb5':[34,72]
            ,'scpc1':[35,227]
            ,'scpc2':[36,219]
            ,'scpc3':[37,243]
            ,'scpc4':[38,219]
            ,'scpc5':[39,215]
            ,'scpd1':[40,60]
            ,'scpd2':[41,66]
            ,'scpd3':[42,72]
            ,'scpd4':[43,62]
            ,'scpd5':[44,61]
            ,'scpnre1':[45,29]
            ,'scpnre2':[46,30]
            ,'scpnre3':[47,27]
            ,'scpnre4':[48,28]
            ,'scpnre5':[49,28]
            ,'scpnrf1':[50,14]
            ,'scpnrf2':[51,15]
            ,'scpnrf3':[52,14]
            ,'scpnrf4':[53,14]
            ,'scpnrf5':[54,13]
            ,'scpnrg1':[55,176]
            ,'scpnrg2':[56,154]
            ,'scpnrg3':[57,166]
            ,'scpnrg4':[58,168]
            ,'scpnrg5':[59,168]
            ,'scpnrh1':[60,63]
            ,'scpnrh2':[61,63]
            ,'scpnrh3':[62,59]
            ,'scpnrh4':[63,58]
            ,'scpnrh5':[64,55]
        }

        for nomInstancia in orden:
            if nomInstancia in archivoInstancia:
                #print(f"instancia {nomInstancia}")
                return orden[nomInstancia][1]

        return None