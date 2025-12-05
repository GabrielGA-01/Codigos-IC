# Iguala a quantidade de rótulos um e rótulos zero entre dois grupos
# base_de_dados - O dataframe Pandas
# colunas_discretas - Uma lista contendo os nomes das colunas
# variavel_sensivel - O nome da coluna que contém esses dois grupos
# classe_alterada - O valor da classe que será igualado as ocorrências de target 1 e target 0
# classe_inalterada - O valor da outra classe
# printar - Determina se háverá prints do processo ou não

import pandas as pd
from sklearn import config_context
from imblearn.over_sampling import SMOTENC
import c0_1_configuracoes as c0_1_configuracoes
seed = c0_1_configuracoes.seed

def enviesamento_smotenc_rotulo_binario(base_de_dados, colunas_discretas, variavel_sensivel, classe_alterada, classe_intacta, printar=False):
  df_alterado = base_de_dados[base_de_dados[variavel_sensivel] == classe_alterada].copy()
  df_mantido = base_de_dados[base_de_dados[variavel_sensivel] == classe_intacta].copy()

  alterado_target1_orig = df_alterado['target'].sum()
  alterado_target0_orig = df_alterado.shape[0] - alterado_target1_orig

  mantido_target1_orig = df_mantido['target'].sum()
  mantido_target0_orig = df_mantido.shape[0] - mantido_target1_orig

  if(printar):
    print("\n----- Dados recebidos -----")
    print(f"Total da classe alterada: {df_alterado.shape[0]}")
    print(f"Total de classe mantida: {df_mantido.shape[0]}")
    print(f"\n----- Target original da classe alterada -----")
    print(f"Total da classe alterada com target = 0: {alterado_target0_orig}")
    print(f"Total da classe alterada com target = 1: {alterado_target1_orig}")
    print(f"\n----- Target original da classe mantida -----")
    print(f"Total de classe mantida com target = 0: {mantido_target0_orig}")
    print(f"Total de classe mantida com target = 1: {mantido_target1_orig}")

  # Aplicar SMOTENC nos conjunto que será alterado

  # Separar features (X) e target (y) para o subgrupo que será alterado
  X_alterado = df_alterado.drop('target', axis=1)
  y_alterado = df_alterado['target']

  # Identificar os índices das colunas categóricas
  categorical_features_indices = [X_alterado.columns.get_loc(col) for col in colunas_discretas]

  # Igualar a quantidade de target = 1 e targer = 0
  if(alterado_target0_orig > alterado_target1_orig):
    estrategia = {1: alterado_target0_orig}
  else:
    estrategia = {0: alterado_target1_orig}

  # Instanciando o SMOTENC
  smotenc = SMOTENC(sampling_strategy=estrategia, categorical_features=categorical_features_indices, random_state=seed)

  # Aplicar o SMOTE
  with config_context(transform_output="default"):
   X_alterado_resampled, y_alterado_resampled = smotenc.fit_resample(X_alterado, y_alterado)

  # Reconstruir o DataFrame com os novos dados sintéticos do grupo alterado
  df_alterado_resampled = pd.concat([
      pd.DataFrame(X_alterado_resampled, columns=X_alterado.columns),
      pd.Series(y_alterado_resampled, name='target')
  ], axis=1)

  # Juntar o grupo inalterado com o novo grupo alterado
  df_enviesado_smotenc = pd.concat([df_mantido, df_alterado_resampled], ignore_index=True)

  alterado_target1_smotenc = df_alterado_resampled['target'].sum()
  alterado_target0_smotenc = df_alterado_resampled.shape[0] - alterado_target1_smotenc

  mantido_target1_smotenc = df_mantido['target'].sum()
  mantido_target0_smotenc = df_mantido.shape[0] - mantido_target1_smotenc

  if(printar):
    print("\n----- Dados após SMOTENC -----")
    print(f"Total da classe alterada: {df_alterado_resampled.shape[0]}")
    print(f"Total de classe mantida: {df_mantido.shape[0]}")
    print(f"\n----- Target original da classe alterada -----")
    print(f"Total da classe alterada com target = 0: {alterado_target0_smotenc}")
    print(f"Total da classe alterada com target = 1: {alterado_target1_smotenc}")
    print(f"\n----- Target original da classe mantida -----")
    print(f"Total de classe mantida com target = 0: {mantido_target0_smotenc}")
    print(f"Total de classe mantida com target = 1: {mantido_target1_smotenc}")

  return(df_enviesado_smotenc)

# Exemplo de uso
if __name__ == "__main__":
  from sklearn.model_selection import train_test_split
  from c0_1_configuracoes import caminho_processado

  colunas_discretas1 = [
      'IS_MARRIAGED', 'EDUCATION', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
  ]

  df = pd.read_csv(f"{caminho_processado}/Dataset1.csv", sep=',')
  X = df.drop('target', axis=1)
  y = df['target']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

  df_treino = pd.concat([X_train, y_train], axis=1)
  df_teste = pd.concat([X_test, y_test], axis=1)

  df_treino_enviesado_smotenc = enviesamento_smotenc_rotulo_binario(df_treino, colunas_discretas1, "sensitive_age", 0, 1, printar=True)