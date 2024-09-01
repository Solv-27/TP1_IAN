import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, JUGADAS_STR

# Clases EstadoDiezMil y AmbienteDiezMil (sin cambios, se incluyen para contexto)

class EstadoDiezMil:
    def __init__(self, puntaje_total, puntaje_turno, dados):
        self.puntaje_total = puntaje_total
        self.puntaje_turno = puntaje_turno
        self.dados = dados

    def actualizar_estado(self, puntaje_tirada, dados_a_tirar):
        self.puntaje_turno += puntaje_tirada
        self.dados = dados_a_tirar

    def fin_turno(self):
        self.puntaje_total += self.puntaje_turno
        self.puntaje_turno = 0

    def __str__(self):
        return f"Estado(puntaje_total={self.puntaje_total}, puntaje_turno={self.puntaje_turno}, dados={self.dados})"

class AmbienteDiezMil:
    def __init__(self):
        self.estado = EstadoDiezMil(0, 0, [1, 2, 3, 4, 5, 6])
    
    def reset(self):
        self.estado = EstadoDiezMil(0, 0, [1, 2, 3, 4, 5, 6])
        return len(self.estado.dados)  # Devolver cantidad de dados como estado inicial

    def step(self, action):
        if action == JUGADA_TIRAR:
            dados_tirados = [np.random.randint(1, 7) for _ in self.estado.dados]
            puntaje, dados_no_usados = puntaje_y_no_usados(dados_tirados)
            
            if puntaje == 0:  # Fallo
                reward = -self.estado.puntaje_turno
                self.estado.fin_turno()
                done = True
            else:
                self.estado.actualizar_estado(puntaje, dados_no_usados)
                reward = puntaje
                done = False
        elif action == JUGADA_PLANTARSE:
            self.estado.fin_turno()
            reward = self.estado.puntaje_turno
            done = True
        else:
            raise ValueError("Acción no válida")

        return len(self.estado.dados), reward, done  # para hacer la tabla de estados por cantidad de dados


class AgenteQLearning:
    def __init__(self, ambiente, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.ambiente = ambiente
        self.q_table = defaultdict(lambda: {JUGADA_TIRAR: 0, JUGADA_PLANTARSE: 0})
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def elegir_accion(self, estado):
        if (self.ambiente.estado.puntaje_total < 750):
            return JUGADA_TIRAR
        else:
            if np.random.rand() < self.epsilon:
                return np.random.choice([JUGADA_TIRAR, JUGADA_PLANTARSE])
            else:
                return max(self.q_table[estado].items(), key=lambda x: x[1])[0]

    def entrenar(self, episodios, verbose=False):
        for episodio in tqdm(range(episodios), desc="Entrenando"):
            estado = self.ambiente.reset()
            done = False
            
            while not done:
                accion = self.elegir_accion(estado)
                nuevo_estado, recompensa, done = self.ambiente.step(accion)
                
                if nuevo_estado not in self.q_table:
                    self.q_table[nuevo_estado] = {JUGADA_TIRAR: 0, JUGADA_PLANTARSE: 0}
                
                mejor_futuro = max(self.q_table[nuevo_estado].values())
                self.q_table[estado][accion] += self.alpha * (recompensa + self.gamma * mejor_futuro - self.q_table[estado][accion])

                estado = nuevo_estado
            
            if verbose:
                print(f"Época {episodio}: Estado {estado}, Acción {accion}, Recompensa {recompensa}")

            if self.epsilon > 0.1:
                self.epsilon *= 0.995


            print(f"Tabla Q después del episodio {episodio}: {dict(self.q_table)}")

    def guardar_politica(self, filename):
        print("Guardando política. Contenido de la tabla Q:")
        with open(filename, "w") as f:
            for estado, acciones in self.q_table.items():
                f.write(f"{estado},{acciones[JUGADA_TIRAR]},{acciones[JUGADA_PLANTARSE]}\n")



class JugadorEntrenado:
    def __init__(self, nombre, politica_filename):
        self.nombre = nombre
        self.q_table = self.cargar_politica(politica_filename)

    def cargar_politica(self, filename):
        q_table = defaultdict(lambda: {JUGADA_TIRAR: 0, JUGADA_PLANTARSE: 0})
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                estado = int(parts[0]) 
                q_table[estado] = {
                    JUGADA_TIRAR: float(parts[1]),  # Valor para JUGADA_TIRAR
                    JUGADA_PLANTARSE: float(parts[2])  # Valor para JUGADA_PLANTARSE
                }
        return q_table

    def elegir_accion(self, estado):
        acciones = self.q_table[estado]  # Estado es simplemente el número de dados restantes
        return max(acciones, key=acciones.get)
    
    def jugar(self, puntaje_total, puntaje_turno, dados):
        estado = len(dados) 

        if puntaje_total <750:
            accion = JUGADA_TIRAR
        else:
            accion = self.elegir_accion(estado)
        
        if accion == JUGADA_TIRAR:
            puntaje_tirada, dados_no_usados = puntaje_y_no_usados(dados)
            #pierde si de la jugo y no sumo puntos o tambien si ya no le quedan dados
            if puntaje_tirada == 0 or len(dados_no_usados) == 0:
                accion = JUGADA_PLANTARSE
                dados_a_tirar = []
            else:
                dados_a_tirar = dados_no_usados  # se tiran los dados que le quedan
        
        elif accion == JUGADA_PLANTARSE:
            dados_a_tirar = []  # ya no tiene dados porque se planto

        return accion, dados_a_tirar

    
