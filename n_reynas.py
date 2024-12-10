import numpy as np
import random

# Función para evaluar el estado: número de conflictos entre reinas
def evaluar(tablero):
    conflictos = 0
    n = len(tablero)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(tablero[i] - tablero[j]) == abs(i - j):  # Conflicto diagonal
                conflictos += 1
    return conflictos

# Generar un vecino aleatorio cambiando una reina de fila
def generar_vecino(tablero):
    n = len(tablero)
    nuevo_tablero = tablero.copy()
    col = random.randint(0, n - 1)
    nueva_fila = random.randint(0, n - 1)
    nuevo_tablero[col] = nueva_fila
    return nuevo_tablero

# Algoritmo de recocido simulado
def recocido_simulado(n, temperatura_inicial, enfriamiento, iteraciones_max):
    # Estado inicial aleatorio
    tablero_actual = np.random.permutation(n)
    conflictos_actual = evaluar(tablero_actual)

    temperatura = temperatura_inicial

    for _ in range(iteraciones_max):
        if conflictos_actual == 0:
            break

        # Generar vecino y evaluar
        tablero_vecino = generar_vecino(tablero_actual)
        conflictos_vecino = evaluar(tablero_vecino)

        # Calcular probabilidad de aceptación
        delta = conflictos_vecino - conflictos_actual
        if delta < 0 or random.uniform(0, 1) < np.exp(-delta / temperatura):
            tablero_actual = tablero_vecino
            conflictos_actual = conflictos_vecino

        # Reducir temperatura
        temperatura *= enfriamiento

    return tablero_actual, conflictos_actual

# Resolver para N=8
n_reinas = 8
solucion, conflictos = recocido_simulado(n_reinas, temperatura_inicial=1000, enfriamiento=0.99, iteraciones_max=10000)
solucion, conflictos
