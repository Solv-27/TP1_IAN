import argparse
from diezmil import JuegoDiezMil
from template import JugadorEntrenado
import numpy as np
import matplotlib.pyplot as plt


def graficar(turnos):

    # Crear el eje x con los números de jugada
    x = np.arange(1, len(turnos) + 1)

    # Calcular la media de los números
    media = np.mean(turnos)
    varianza = np.var(turnos)
    print(media)
    print(varianza)

    plt.scatter(x, turnos, color='turquoise')
    # Añadir una línea horizontal en la media
    plt.axhline(y=media, color='salmon', linestyle='--', label=f'Media: {media:.2f}')

    plt.title('Rendimiento de agente en cantidad de turnos')
    plt.xlabel('Número de Jugada')
    plt.ylabel('Cantidad de Turnos')

    plt.show()

def main(politica_filename, verbose):
    politica_filename = 'politica.csv'
    jugador = JugadorEntrenado('qlearning', politica_filename)
    juego = JuegoDiezMil(jugador)
    turnos_totales = []
    #Para poder medir el rendimiento de los hiperparámetros propuestos, hacemos jugar a nuestro agente x veces
    #Ya que la cantidad de turnos que se toma suele variar, vamos a graficarlos para ver su media
    for i in range(250):
        cantidad_turnos, puntaje_final = juego.jugar(verbose=verbose)
        turnos_totales.append(cantidad_turnos)

        #Imprimo la cantidad de puntos y turnos totales por jugada
        print(f"Cantidad de turnos: {cantidad_turnos}")
        print(f"Puntaje final: {puntaje_final}")

    graficar(turnos_totales)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jugar una partida de 'Diez Mil' con un agente entrenado usando una política predefinida.")

    # Agregar argumentos
    parser.add_argument('-f', '--politica_filename', type=str, help='Archivo con la política entrenada')
    parser.add_argument('-v', '--verbose', action='store_true', help='Activar modo verbose para ver más detalles durante el juego')

    # Parsear los argumentos
    args = parser.parse_args()

    # Llamar a la función principal con los argumentos proporcionados
    main(args.politica_filename, args.verbose)
