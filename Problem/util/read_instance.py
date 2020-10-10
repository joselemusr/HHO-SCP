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

        Archivo = open(Instancia, "r")

        # Leer Dimensión
        Registro = Archivo.readline().split()
        self.rows = int(Registro[0])
        self.columns      = int(Registro[1])
        
        # Leer Costo
        Costos        = []
        Registro      = Archivo.readline()
        ContVariables = 1
        while Registro != "" and ContVariables <= self.columns :
            Valores = Registro.split()
            for Contador in range(len(Valores)):
                Costos.append(int(Valores[Contador]))
                ContVariables = ContVariables + 1
            Registro = Archivo.readline()

        # Preparar Matriz de Restricciones
        Restricciones = np.zeros((self.rows,self.columns), dtype=np.int32).tolist()

        # Leer Restricciones    
        ContVariables      = 1
        Fila               = 0
        cont = 0
        while Registro != "":
            CantidadValoresUno = int(Registro)
            ContadorValoresUno = 0
            Registro = Archivo.readline()
            Registro = Registro.replace('\n','').replace("\\n'",'')
            while Registro != "" and ContadorValoresUno < CantidadValoresUno: 
                Columnas = Registro.split() 
                for Contador in range(len(Columnas)):
                    Columna = int(Columnas[Contador]) - 1
                    Restricciones[Fila][Columna] = 1
                    ContadorValoresUno = ContadorValoresUno + 1
                Registro = Archivo.readline()
            Fila = Fila + 1

        Archivo.close()
        self.set_c(Costos)
        self.set_r(Restricciones)        