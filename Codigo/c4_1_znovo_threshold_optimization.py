"""
Este arquivo implementa a técnica de pós-processamento Threshold Optimization (Fairlearn).
Ela ajusta os limiares de decisão de um classificador já treinado para otimizar a métrica de justiça Demographic Parity.
"""

from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split

from c0_1_configuracoes import seed
from c1_4_desempenho_modelos import desempenho_modelo

import numpy as np
import random

# parametros_dataset_pos: dados sensíveis e printar testes
# dados_dataset: dados de treino, validação e teste e o nome do caso de teste
# modelo: modelo já treinado

# Obs.: parametros_algoritmos não é mais necessário aqui. Será uma informação usada externa à função

def threshold_optimization(parametros_dataset_pos, dados_dataset, modelo, nome_modelo):
    # Para não alterar os dados originais (os dados de treino, validação e teste não são alterados aqui)
    nome_do_teste = dados_dataset['nome_base_de_dados'] + f" || THRESHOLD OPTIMIZATION || {nome_modelo.upper().replace('_', ' ')}"

    # Criando uma variável para os dados sensíveis e coluna sensível
    dados_sensiveis = parametros_dataset_pos['dados_sensiveis']
    coluna_sensivel = dados_sensiveis['coluna_sensivel']

    X_train = dados_dataset['X_train']

    X_val = dados_dataset['X_val']
    y_val = dados_dataset['y_val']

    X_test = dados_dataset['X_test']
    y_test = dados_dataset['y_test']

    # Atributo pessoal
    A_val = X_val[coluna_sensivel]
    A_test = X_test[coluna_sensivel]

    # Garantir a replicabilidade
    np.random.seed(seed)
    random.seed(seed)

    optimizer = ThresholdOptimizer(estimator=modelo, constraints='demographic_parity', prefit=True)
    optimizer.fit(X_val, y_val, sensitive_features=A_val)
    y_pred_threshold_optimization = optimizer.predict(X_test, sensitive_features=A_test)

    desempenho_threshold_optimization = desempenho_modelo(nome_do_teste, modelo, nome_modelo, X_train, X_test, X_test, y_pred_threshold_optimization, y_test, dados_sensiveis, printar=parametros_dataset_pos['printar'])

    return(desempenho_threshold_optimization)





if __name__ == "__main__":
    import pandas as pd

    from c0_1_configuracoes import (
        param_grid_random_forest_basico,
        param_grid_xgboost_basico,
        param_grid_regressao_logistica_basico,
        param_grid_perceptron_basico,
        preprocessor_passthrough,
        dados_sensiveis_sexo,
        dados_sensiveis_age,
        )
    from c1_3_preprocessor_automatizado import aplicar_pre_processador
    from c2_1_random_forest import random_forest_GSCV
    from c2_2_xgboost import xgboost_GSCV
    from c2_3_regressao_logistica import regressao_logistica_GSCV
    from c2_4_perceptron import perceptron_GSCV

    df = pd.read_csv(f"Datasets/Processados/Dataset3.csv", sep=',')

    # Pré-processando e separando os dados em 70% treino e 30% teste
    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    # Separando os dados de treino (70%*) em treino (55%*) e validação (15%*) *Em relação ao total de dados
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15/70, random_state=seed, stratify=y_train)

    dados_sensiveis = dados_sensiveis_age

    parametros_dataset_pos = {}
    parametros_dataset_pos['printar'] = True
    parametros_dataset_pos['dados_sensiveis'] = dados_sensiveis
    
    dados_dataset = {}
    dados_dataset['X_train'] = X_train
    dados_dataset['X_val'] = X_val
    dados_dataset['X_test'] = X_test
    dados_dataset['y_train'] = y_train
    dados_dataset['y_val'] = y_val
    dados_dataset['y_test'] = y_test
    dados_dataset['nome_base_de_dados'] = "DATASET"

    # Treinando um modelo

    algoritmo = 2
    if algoritmo == 1:
        nome_modelo = 'random_forest'
        funcao = random_forest_GSCV
        param_grid = param_grid_random_forest_basico
    elif algoritmo == 2:
        nome_modelo = 'xgboost'
        funcao = xgboost_GSCV
        param_grid = param_grid_xgboost_basico
    elif algoritmo == 3:
        nome_modelo = 'regressao_logistica'
        funcao = regressao_logistica_GSCV
        param_grid = param_grid_regressao_logistica_basico
    elif algoritmo == 4:
        nome_modelo = 'perceptron'
        funcao = perceptron_GSCV
        param_grid = param_grid_perceptron_basico
    
    parametros = {
        'param_grid': param_grid,
        'preprocessor': preprocessor_passthrough,
        'cv_n_splits': 2,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'nome_base_de_dados': "DATASET",
        'dados_sensiveis': dados_sensiveis
    }

    modelo, _ = funcao(**parametros)

    desempenho = threshold_optimization(parametros_dataset_pos, dados_dataset, modelo, nome_modelo)