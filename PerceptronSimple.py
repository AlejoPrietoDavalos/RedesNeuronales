import numpy as np

class PerceptronSimple():
    """ Perceptrón simple con N capas de entrada, M capas de salida y un ratio de aprendizaje de "eta".
    
    Args:
        N (int): Número de capas de entrada.
        M (int): Número de capas de salida.
        eta (float): Ratio de aprendizaje.
        bias (float): Bias o sesgo.
        f_act (function): Función de activación."""
    def __init__(self, N: int, M: int, eta: float, bias: float, f_act):
        self.N = N                                          # Capas de entrada. Recordar que sería N+1 por que sumamos el bias.
        self.M = M                                          # Capas de salida.
        self.eta = eta                                      # Ratio de aprendizaje.
        self.bias = bias                                    # Bias
        self.f_act = f_act                                  # Función de activación
        self.W = self.CrearW(N, M)                          # Matriz de pesos. W[i][j] contiene el peso asociado a la conexión entre la neurona de entrada "i" con la neurona de salida "j".

    def Entrenar(self, X: np.ndarray, Z: np.ndarray) -> None:
        for X_h, Z_h in zip(X, Z):                          # Recorremos la matriz de datos de prueba y el resultado esperado. La "h" es por la nomenclatura que usa el profesor.
            Y_h = self.Prediccion(X_h)
            E_h = self.Error(Z_h, Y_h)
            self.Correccion(X_h, E_h)

    def Correccion(self, X_h: np.ndarray, E_h: np.ndarray) -> None:
        """Recibe el h-ésimo dato de X y el error obtenido en la predicción, corrije los pesos W.
        
        Args:
            X_h (array): Dato h-ésimo del conjunto de set de entrenamiento.
            E_h (array): Vector de diferencias."""
        X_h, E_h = np.array([np.concatenate([X_h, [self.bias]])]), np.array([E_h])         # Esto es para poder tener 2 vectores con 1 fila. Por alguna razón no me deja hacer el producto escalar por no tener "1 fila".

        d_W = self.eta * np.dot(X_h.T, E_h)
        self.W = self.W + d_W                               # Hacemos la corrección del modelo.

    def Error(self, Z_h: np.ndarray, Y_h: np.ndarray) -> np.ndarray:
        """ Retorna el vector de diferencias con el error cometido por aplicar este modelo.
        Resultado de hacer la diferencia instancia por instancia entre el resultado esperado, y el obtenido.
        
        Args:
            Z_h (array): Resultado esperado del modelo para el dato h-ésimo de X.
            Y_h (array): Resultado obtenido del modelo para el dato h-ésimo de X.
        return:
            E_h (array): Vector de diferencias."""
        E_h = Z_h - Y_h
        return E_h

    def Prediccion(self, X_h: np.ndarray):
        """ Devuelve la predicción del modelo. Para el h-ésimo dato.
        
        Args:
            X_h (array): Dato h-ésimo del conjunto de set de entrenamiento.
        Return:
            Y_h (array): Predicción del modelo."""
        X_h = np.concatenate([X_h,[self.bias]])
        Y_h = self.f_act(np.dot(X_h, self.W))
        return Y_h

    def CrearW(self, N: int, M: int) -> np.ndarray:
        """ Crea la matriz de pesos W de tamaño N+1xM, cada peso está inicializado al azar
        con una distribución normal centrada en 0 y varianza '1/np.sqrt(N)'.
        La matriz W(i,j) contiene el peso de la neurona de entrada i conectada a la neurona de salida j.

        Args:
            N (int): Número de capas de entrada.
            M (int): Número de capas de salida.
        Return:
            W (array): Matriz de pesos."""
        W = np.array([np.random.normal(0, 1/np.sqrt(N), M) for _ in range(N+1)])
        return W