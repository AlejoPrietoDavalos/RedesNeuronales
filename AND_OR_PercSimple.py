""" Vamos a crear un perceptrón simple que simule los condicionales AND y OR.
Debe tener como entra 2 neuronas para las proposiciones, y 2 neuronas de salida para el resultado del AND y del OR.
La posición 0 es el resultado del AND y la posición 1 del OR en el vector de resultados."""

from PerceptronSimple import PerceptronSimple
import numpy as np
import random

red = PerceptronSimple(2, 2, 0.1, -1, np.sign)
n_datos = 100

# Vamos a hacer el AND y el OR juntos.
X = np.array( [[random.randint(0,1), random.randint(0,1)] for _ in range(n_datos)] )
Z = []      # La primera es AND y la segunda OR.
for dato in X:
    if dato[0]==0 and dato[1]==0:
        Z.append([-1,-1])
    elif dato[0]==0 and dato[1]==1:
        Z.append([-1,1])
    elif dato[0]==1 and dato[1]==0:
        Z.append([-1,1])
    else:
        Z.append([1,1])
Z = np.array(Z)



red.Entrenar(X, Z)

while True:
    a = input()
    b = input()
    print(red.Prediccion(np.array([int(a),int(b)])))