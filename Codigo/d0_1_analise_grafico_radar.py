from itertools import permutations
import math

def encontrar_todas_permutacoes(vetor):
    print(f"Permutações para o vetor: {vetor}\n")
    print("Utiliza-se os angulos em radianos")
    perms = permutations(vetor, len(vetor))

    # Calcula a área para cada permutação
    resultados = []
    for perm in perms:
        print(perm)
        soma = 0
        for x in range(len(vetor)):
            soma += perm[x-1]*perm[x]/2*math.sin((2 * math.pi) /len(vetor))
            print(f"({perm[x]} * {perm[x-1]} / 2) * sen({(2 * math.pi) /len(vetor):.3f}) = {perm[x-1]*perm[x]/2*math.sin((2 * math.pi) /len(vetor)):.3f}")
        print(f"A soma das áreas é {soma:.3f}\n")
        resultados.append(soma)

    resultados.sort()
    for x in resultados:
        print(f"{x:.4f}", end=", ")

if __name__ == "__main__":
    encontrar_todas_permutacoes([0.9, 0.8, 0.5, 0.6])
    # encontrar_todas_permutacoes([0.9, 0.9, 0.5, 0.3, 0.3])
    # encontrar_todas_permutacoes([0.8, 0.7, 0.9, 1, 0.6])