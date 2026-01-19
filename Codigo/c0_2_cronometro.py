"""
Este arquivo contém utilitários para medição de tempo de execução.
"""

import time
from contextlib import contextmanager

@contextmanager
def cronometro():
    """
    Gerenciador de contexto para medir o tempo de parede (wall time) e o tempo de CPU.
    
    Variáveis internas:
    - start_wall: Armazena o tempo de início usando time.perf_counter() (tempo real decorrido).
    - start_cpu: Armazena o tempo de início usando time.process_time() (tempo de CPU consumido pelo processo).
    
    Retorna:
    - Uma função lambda que, quando chamada, retorna um dicionário com:
        - 'total': Diferença entre o tempo de parede atual e o inicial.
        - 'cpu': Diferença entre o tempo de CPU atual e o inicial.
    """
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    yield lambda: {
        'total': time.perf_counter() - start_wall, 
        'cpu': time.process_time() - start_cpu
    }