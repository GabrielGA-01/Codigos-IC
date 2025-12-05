# Função utilizada para garantir que em um determinado conjunto de dados
# tenha a mesma proporção de intâncias com o rótulo sendo zero e com o rótulo sendo um
# conjunto_dados - Dataframe Pandas
# conjunto_nome - Um nome para se referir a esse conjunto. Impacta somente no print
# rotulo - O nome da coluna de rótulo
# printar - Determina se háverá prints do processo ou não

def balancear_proporcao_rotulo_binario(conjunto_dados, cojunto_nome, rotulo, printar=False):
  estatisticas = {}

  # Proporção do rótulo
  proporcao = conjunto_dados[rotulo].value_counts(normalize=True)
  estatisticas["proporcao_inicial"] = proporcao
  if(printar):
    print(f"\n----- {cojunto_nome} || PROPORÇÃO INICIAL DAS CLASSES DO RÓTULO -----\n")
    print(proporcao)

  # Verificando de cada classe do rótulo
  if(printar):
    print(f"\n----- {cojunto_nome} || TOTAL INICIAL DE CADA CLASSE DO RÓTULO -----\n")

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
    print(f"\n----- {cojunto_nome} BALANCEADO || PROPORÇÃO FINAL DAS CLASSES DO RÓTULO -----\n")
    print(proporcao)

  # Verificando de cada classe do rótulo
  if(printar):
    print(f"\n----- {cojunto_nome} BALANCEADO || TOTAL FINAL DE CADA CLASSE DO RÓTULO -----\n")

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