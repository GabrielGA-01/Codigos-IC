from c3_1_novo_instance_reweighing import instance_reweighing  # É preciso começar por ele para não dar problema com o torch vindo da aif360

# Bibliotecas
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import traceback

# Variáveis auxiliares
from c0_1_configuracoes import(
  param_grid_perceptron_basico,
  param_grid_random_forest_basico,
  param_grid_regressao_logistica_basico,
  param_grid_xgboost_basico,
  param_grid_perceptron_completo,
  param_grid_random_forest_completo,
  param_grid_regressao_logistica_completo,
  param_grid_xgboost_completo,
  preprocessor_passthrough,
  seed
)

from c0_2_cronometro import cronometro

# Funções auxiliares
from c1_6_enviesamento import enviesar

# Algoritmos
from c2_1_random_forest import random_forest_GSCV
from c2_2_xgboost import xgboost_GSCV
from c2_3_regressao_logistica import regressao_logistica_GSCV
from c2_4_perceptron import perceptron_GSCV

# Técnicas de pré-processamento
from c3_1_novo_instance_reweighing import instance_reweighing
from c3_2_novo_disparate_impact_removal import disparate_impact_removal
from c3_3_znovo_synthetic_data_generation import synthetic_data_generation
from c3_4_znovo_suppression import suppression

# Técnicas de pós-processamento
from c4_1_znovo_threshold_optimization import threshold_optimization
from c4_2_novo_calibration import calibration
from c4_3_znovo_reject_option_classification import reject_option_classification

# Interpretabilidade
from c5_1_interpretabilidade import gerar_interpretabilidade, gerar_interpretabilidade_especifica

# Gerar e interpretar resultados
from c6_1_znovo_salvar_resultados import gerar_planilha_nova as gerar_planilha, salvar_dicionario

# Garantir a replicabilidade
np.random.seed(seed)
random.seed(seed)

caminho_resultado = './Resultados'

# Chama a função que aplica todos os enviesamento considerando as variáveis sensiveis
# Não compensa salvar o arquivo pois seria muito pesado e leva apenas ~10 segundos para executar
datasets = enviesar()

# Removendo os originais dos datasets 1 e 2 | Removendo os smotes dos datasets 3 e 4

del datasets['df1']['original_sensitive_sexo']
del datasets['df1']['original_sensitive_age']
del datasets['df2']['original_sensitive_sexo']
del datasets['df2']['original_sensitive_age']
del datasets['df3']['smote_simples_sensitive_sexo']
del datasets['df3']['smote_simples_sensitive_age']
del datasets['df4']['smote_simples_sensitive_sexo']
del datasets['df4']['smote_simples_sensitive_age']

modo_completo = 1

param_grid_random_forest = param_grid_random_forest_completo if modo_completo else param_grid_random_forest_basico
param_grid_perceptron = param_grid_perceptron_completo if modo_completo else param_grid_perceptron_basico
param_grid_regressao_logistica = param_grid_regressao_logistica_completo if modo_completo else param_grid_regressao_logistica_basico
param_grid_xgboost = param_grid_xgboost_completo if modo_completo else param_grid_xgboost_basico

parametros_algoritmos = {
    'random_forest': {
        'funcao': random_forest_GSCV,
        'parametros_grid': param_grid_random_forest
    },
    'xgboost': {
        'funcao': xgboost_GSCV,
        'parametros_grid': param_grid_xgboost
    },
    'regressao_logistica': {
        'funcao': regressao_logistica_GSCV,
        'parametros_grid': param_grid_regressao_logistica
    },
    'perceptron': {
        'funcao': perceptron_GSCV,
        'parametros_grid': param_grid_perceptron
    }
}

printar_tecnicas = False

# Parâmetros comuns para todos
parametros_gerais = {
  'printar': True,
  'cv_n_splits': 2
}

resultado_global = {}
sucesso = 0

# Percorre cada base de dados distinta
try:

  for df in datasets:

    print(f"\n\n=== INICIANDO ANÁLISE DO {df.upper()} ===\n")
    # Percorre cada tipo de enviesamento
    for tipo in datasets[df]:

      print(f"\n--- Analisando o tipo: {tipo} ---\n")

      banco = datasets[df][tipo]
      nome_banco = banco['nome_banco']

      # Separando dados de treino e teste
      X_train = banco['treino'].drop('target',axis=1)
      y_train = banco['treino']['target']

      X_test = banco['teste'].drop('target', axis=1)
      y_test = banco['teste']['target']

      # Formando os parâmetros para as funções
      parametros_dataset = {
        'preprocessor': preprocessor_passthrough,
        'dados_sensiveis': banco['dados_sensiveis']
      }
      parametros_dataset.update(parametros_gerais)

      dados_dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'nome_base_de_dados': nome_banco
      }

      resultado_dataset = {}
      resultado_dataset['smote'] = banco['smote']   # Tipo de smote usado no dataset em questão

      # Aplicando as técnicas de pré-processamento
      resultado_dataset['instance_reweighing']        = instance_reweighing         (parametros_dataset, dados_dataset, parametros_algoritmos, printar_tecnicas)
      resultado_dataset['disparate_impact_removal']   = disparate_impact_removal    (parametros_dataset, dados_dataset, parametros_algoritmos, printar_tecnicas)
      resultado_dataset['synthetic_data_generation']  = synthetic_data_generation   (parametros_dataset, dados_dataset, parametros_algoritmos, banco['colunas_discretas'],printar_tecnicas)
      resultado_dataset['suppression']                = suppression                 (parametros_dataset, dados_dataset, parametros_algoritmos, printar_tecnicas)

      # Aplicando as técnicas de pós-processamento + sem técnica
      resultado_dataset['threshold_optimization'] = {}
      resultado_dataset['calibration'] = {}
      resultado_dataset['reject_option_classification'] = {}
      resultado_dataset['sem_tecnica'] = {}

      # Separando os dados de treino (70%*) em treino (55%*) e validação (15%*) *Em relação ao total de dados
      X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split(X_train, y_train, test_size=15/70, random_state=seed, stratify=y_train)

      # Atualizando os dados com a nova divisão. Dados de teste e o nome da base de dados continuam iguais
      dados_dataset_pos = dados_dataset.copy()
      dados_dataset_pos['X_train'] = X_train_pos
      dados_dataset_pos['y_train'] = y_train_pos

      # Parametros para as técnicas de pós-processamento
      parametros_modelo_pos = parametros_dataset | dados_dataset_pos

      # Adicionando os dados de validação para as técnicas
      dados_dataset_pos['X_val'] = X_val_pos
      dados_dataset_pos['y_val'] = y_val_pos

      # Parâmetros para o modelo sem técnica
      parametros_modelo_sem_tecnica = parametros_dataset | dados_dataset
      parametros_modelo_sem_tecnica['nome_base_de_dados'] = parametros_modelo_sem_tecnica['nome_base_de_dados'] + " || SEM TÉCNICA"

      for nome_algoritmo in parametros_algoritmos:
        algoritmo = parametros_algoritmos[nome_algoritmo]

        # Modelo sem técnica
        parametros_modelo_sem_tecnica['param_grid'] = algoritmo['parametros_grid']
        _, resultado_dataset['sem_tecnica'][nome_algoritmo] = algoritmo['funcao'](**parametros_modelo_sem_tecnica)

        # Treinando o modelo base para as técnicas de pós-processamento
        parametros_modelo_pos['param_grid'] = algoritmo['parametros_grid']
        modelo, _ = algoritmo['funcao'](**parametros_modelo_pos)

        # Aplicando as técnicas de pós-processamento
        resultado_dataset['threshold_optimization'][nome_algoritmo] = threshold_optimization        (parametros_dataset, dados_dataset_pos, modelo, nome_algoritmo)
        resultado_dataset['calibration'][nome_algoritmo]            = calibration                   (parametros_dataset, dados_dataset_pos, modelo, nome_algoritmo)
        resultado_dataset['reject_option_classification']           = reject_option_classification  (parametros_dataset, dados_dataset_pos, modelo, nome_algoritmo)

      resultado_global[f"{nome_banco.lower().replace(' ', '_')}"] = resultado_dataset.copy()

except KeyboardInterrupt:
  traceback.print_exc()

except Exception:
  traceback.print_exc()

else:
  sucesso = 1

finally:
  print("\n\n\n-------------------------------------------")
  salvar_dicionario(resultado_global, caminho_resultado, sucesso)
  gerar_planilha(resultado_global, caminho_resultado)
  print("-------------------------------------------\n\n\n")