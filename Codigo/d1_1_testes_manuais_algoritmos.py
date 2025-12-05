from c0_1_configuracoes import (
    param_grid_regressao_logistica_completo, 
    param_grid_perceptron_completo, 
    param_grid_random_forest_completo, 
    param_grid_xgboost_completo,
    preprocessor_passthrough,
    dados_sensiveis_sexo,
    dados_sensiveis_age
    )
from c1_3_preprocessor_automatizado import aplicar_pre_processador
from c2_1_random_forest import random_forest_GSCV
from c2_2_xgboost import xgboost_GSCV
from c2_3_regressao_logistica import regressao_logistica_GSCV
from c2_4_perceptron import perceptron_GSCV

import pandas as pd

if __name__ == "__main__":
  # INICIO DA SEÇÃO DE OPÇÕES MANUAIS

  funcao_modelo = regressao_logistica_GSCV
  # funcao_modelo = perceptron_GSCV
  # funcao_modelo = random_forest_GSCV
  # funcao_modelo = xgboost_GSCV

  base_de_dados = pd.read_csv(f"Datasets/Processados/Dataset1.csv", sep=',')
  # base_de_dados = pd.read_csv(f"Datasets/Processados/Dataset2.csv", sep=',')
  # base_de_dados = pd.read_csv(f"Datasets/Processados/Dataset3.csv", sep=',')
  # base_de_dados = pd.read_csv(f"Datasets/Processados/Dataset4.csv", sep=',')

  dados_sensiveis = dados_sensiveis_sexo
  # dados_sensiveis = dados_sensiveis_age

  # FIM DA SEÇÃO DE OPÇÕES MANUAIS

  # Separação de dados e pré-processador
  X_train, X_test, y_train, y_test = aplicar_pre_processador(base_de_dados)

  parametros_grid = None
  if   funcao_modelo == regressao_logistica_GSCV:
      parametros_grid = param_grid_regressao_logistica_completo
  elif funcao_modelo == perceptron_GSCV:
      parametros_grid = param_grid_perceptron_completo
  elif funcao_modelo == random_forest_GSCV:
      parametros_grid = param_grid_random_forest_completo
  elif funcao_modelo == xgboost_GSCV:
      parametros_grid = param_grid_xgboost_completo

  # Parâmetros gerais
  funcao_parametros = {}
  funcao_parametros['param_grid'] = parametros_grid
  funcao_parametros['preprocessor'] = preprocessor_passthrough
  funcao_parametros['cv_n_splits'] = 5
  funcao_parametros['nome_base_de_dados'] = "Dataset"
  funcao_parametros['X_train'] = X_train
  funcao_parametros['X_test'] = X_test
  funcao_parametros['y_train'] = y_train
  funcao_parametros['y_test'] = y_test
  funcao_parametros['dados_sensiveis'] = dados_sensiveis
  funcao_parametros['printar'] = True
  funcao_parametros['matriz_de_confusao'] = False # Não funciona bem em arquivo .py
  funcao_parametros['grafico_shap'] = False

  _, desempenho = funcao_modelo(**funcao_parametros)
  print(desempenho["melhores_parametros"])