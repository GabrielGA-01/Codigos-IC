"""
Este arquivo contém funções para balancear a proporção de classes em um rótulo binário,
garantindo que o número de instâncias de cada classe seja igual (50/50).
"""

import pandas as pd

def balancear_proporcao_rotulo_binario(conjunto_dados, conjunto_nome, rotulo, printar=False):
  """
  Realiza o balanceamento de um dataset binário através de undersampling da classe majoritária.
  
  Parâmetros:
  - conjunto_dados: DataFrame Pandas contendo os dados a serem balanceados.
  - conjunto_nome: String representando o nome do conjunto (ex: 'Treino') para exibição em logs.
  - rotulo: String com o nome da coluna alvo (target).
  - printar: Booleano para definir se as estatísticas do processo devem ser exibidas.
  
  Retorna:
  - df_balanceado: DataFrame Pandas com as classes balanceadas.
  - estatisticas: Dicionário contendo informações sobre a proporção inicial, final e quantidade de linhas removidas.
  """
  estatisticas = {}

  # Proporção do rótulo
  proporcao = conjunto_dados[rotulo].value_counts(normalize=True)
  estatisticas["proporcao_inicial"] = proporcao
  if(printar):
    print(f"\n----- {conjunto_nome} || PROPORÇÃO INICIAL DAS CLASSES DO RÓTULO -----\n")
    print(proporcao)

  # Verificando de cada classe do rótulo
  if(printar):
    print(f"\n----- {conjunto_nome} || TOTAL INICIAL DE CADA CLASSE DO RÓTULO -----\n")

  quantidade_de_cada_classe = {}
  classes = conjunto_dados[rotulo].unique()
  classes.sort()
  for x in classes:
    quantidade = conjunto_dados.loc[conjunto_dados[rotulo] == x].shape[0]
    quantidade_de_cada_classe[x] = quantidade
    if(printar):
      print(f"{rotulo} = {x}: {quantidade}")
  estatisticas["quantidade_inicial"] = quantidade_de_cada_classe

  # Preparando os dados balanceados
  df_balanceado = conjunto_dados.copy()

  # Identificar a maior classe e o total
  max = 0
  total = 0
  for z in classes:
    quantidade = conjunto_dados.loc[conjunto_dados[rotulo] == z].shape[0]
    total += quantidade
    if quantidade > max:
      max = quantidade
      classe_max = z
  indices_remover = df_balanceado[df_balanceado[rotulo] == classe_max].index

  # Determinar quanto deve ser removido para balancear os dados
  n_remover = int(len(indices_remover) * (1 - (total-len(indices_remover))/len(indices_remover)))

  if(printar):
    print(f"Total de linhas {rotulo} = {classe_max}: {len(indices_remover)}")
    print(f"Número de linhas a serem removidas: {n_remover}\n")

  # Separar aleatoriamente os indices para serem removidos
  indices_para_remover = pd.Series(indices_remover).sample(n=n_remover, random_state=42)
  # Remover os casos selecionados do dataset principal
  df_balanceado = df_balanceado.drop(indices_para_remover)
  estatisticas["classe_reduzida"] = classe_max
  estatisticas["linhas_removidas"] = n_remover

  # Proporção do rótulo final
  proporcao = df_balanceado[rotulo].value_counts(normalize=True)
  estatisticas["proporcao_final"] = proporcao
  if(printar):
    print(f"\n----- {conjunto_nome} BALANCEADO || PROPORÇÃO FINAL DAS CLASSES DO RÓTULO -----\n")
    print(proporcao)

  # Verificando de cada classe do rótulo
  if(printar):
    print(f"\n----- {conjunto_nome} BALANCEADO || TOTAL FINAL DE CADA CLASSE DO RÓTULO -----\n")

  quantidade_de_cada_classe = {}
  classes = df_balanceado[rotulo].unique()
  classes.sort()
  for x in classes:
    quantidade = df_balanceado.loc[df_balanceado[rotulo] == x].shape[0]
    quantidade_de_cada_classe[x] = quantidade
    if(printar):
      print(f"{rotulo} = {x}: {quantidade}")
  estatisticas["quantidade_final"] = quantidade_de_cada_classe

  return(df_balanceado, estatisticas)

# Exemplo de uso
if __name__ == "__main__":
  import pandas as pd
  from c0_1_configuracoes import caminho_processado

  df = pd.read_csv(f"{caminho_processado}/Dataset1.csv", sep=',')
  df_original_balanceado, estatisticas = balancear_proporcao_rotulo_binario(df, "dataset1", "target", printar=True)