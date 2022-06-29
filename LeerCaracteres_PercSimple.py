from PerceptronSimple import PerceptronSimple
import numpy as np
import pandas as pd
import random

class PerceptronSimple2(PerceptronSimple):      # La función np.heaviside "escalonada", necesita el segundo argumento.
    def Prediccion(self, X_h: np.ndarray):

        X_h = np.concatenate([X_h,[self.bias]])
        Y_h = self.f_act(np.dot(X_h, self.W), 1)
        return Y_h

class Caracteres():
    def __init__(self):
        """ Estos son todos los caracteres que va a reconocer la red neuronal, en el mismo orden que están en el excel.
            Notar: La posición del caracter en ese string, determina el índice de la matriz en el excel."""
        self.caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.excel = pd.read_excel("Caracteres 5x5.xlsx")

    def FilasCaracterEnExcel(self, char):
        ind = self.caracteres.index(char)
        fila_i, fila_f = ind*5, (ind+1)*5     # El *5 es por que son matrices de 5x5 pixeles.
        return [fila_i, fila_f]
    
    def ArrayPixeles(self, char):
        """ Dado un caracter, devuelve el array de pixeles con 0 y 1 acorde al excel."""
        filas = self.FilasCaracterEnExcel(char)
        array = []
        for i in range(5):
            array += list(self.excel[i][filas[0]:filas[1]])
        return np.array(array)


n_pixeles = 25
n_letras = 36

red = PerceptronSimple2(n_pixeles, n_letras, 0.1, 0, np.heaviside)     # Tenemos 25 pixeles de entrada. 36 posibilidades en la capa de salida, 1 por cada letra o número.
chars = Caracteres()


nDatosPorCaracter = 50                  # Vamos a pasarle a la red neuronal esta cantidad de veces cada caracter.
listaOrdenCaracter = []
for char in chars.caracteres:
    for _ in range(nDatosPorCaracter):
        listaOrdenCaracter.append(char)
random.shuffle(listaOrdenCaracter)      # Desordeno la lista para que la red aprenda de todos los caracteres de forma aleatoria.



for j in range(nDatosPorCaracter):
    X = []                              # Matriz de datos.
    Z = []                              # Matriz de resultados.
    for c in listaOrdenCaracter[j*len(chars.caracteres) : (j+1)*len(chars.caracteres)]:     # Vamos a mandar los datos de entrenamiento en grupos de 36.
        X.append(chars.ArrayPixeles(c))
        res = np.zeros(n_letras)
        res[chars.caracteres.index(c)] = 1                  # Ponemos un 1, en la posición que le corresponde al caracter correcto.
        Z.append(res)
    red.Entrenar(np.array(X), np.array(Z))


# Comprobamos para cada uno de los posibles caracteres, si el modelo reconoce bien cada caracter.
# En caso de que no sea así, manda un mensaje.
for c in chars.caracteres:
    prediccion = list(red.Prediccion(np.array(chars.ArrayPixeles(c))))
    cant_apariciones_1 = prediccion.count(1)                # Contamos cuantas veces aparece el número 1. Debería ser 1 sola vez, en la posición que corresponde al caracter.
    if cant_apariciones_1 != 1:         # Si la cantidad de apariciones del 1, es distinto de 1. El modelo no es correcto.
        print(f"La letra '{c}', aparece '{cant_apariciones_1}' el numero 1 en el array de resultados.")
    else:
        indice = prediccion.index(1)
        if chars.caracteres[indice] != c:   # Si predice un caracter, pero éste no corresponde con el resultado esperado
            print(f"Para la letra '{c}' la predicción es '{chars.caracteres[indice]}' incorrecta.")
    