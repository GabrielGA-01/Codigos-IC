import time
from contextlib import contextmanager

@contextmanager
# Função para controlar o tempo decorrido
def cronometro():
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    yield lambda: {
        'total': time.perf_counter() - start_wall, 
        'cpu': time.process_time() - start_cpu
    }