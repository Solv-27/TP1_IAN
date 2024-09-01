import argparse
from template import AmbienteDiezMil, AgenteQLearning

def main():
    # Configuración de los argumentos para ejecutar el script
    parser = argparse.ArgumentParser(description="Entrenamiento del agente Q-Learning para el juego Diez Mil")
    parser.add_argument("--episodios", type=int, default=30000, help="Número de episodios de entrenamiento")
    parser.add_argument("--alpha", type=float, default=0.1, help="Tasa de aprendizaje (alpha)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Factor de descuento (gamma)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Probabilidad de exploración inicial (epsilon)")
    parser.add_argument("--verbose", action="store_true", help="Mostrar detalles del entrenamiento")
    parser.add_argument("--guardar", type=str, help="Archivo para guardar la política del agente después del entrenamiento")

    args = parser.parse_args()

    # Crear el ambiente y el agente con los parámetros especificados
    ambiente = AmbienteDiezMil()
    agente = AgenteQLearning(ambiente, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    # Entrenar el agente
    agente.entrenar(episodios=args.episodios, verbose=args.verbose)

    # Guardar la política si se especificó un archivo
    agente.guardar_politica("politica.csv")
    print(f"Política guardada en politica.csv")

if __name__ == "__main__":
    main()
