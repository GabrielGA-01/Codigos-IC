"""
Este arquivo implementa a técnica de Supressão.
Consiste em remover a variável sensível do conjunto de dados antes do treinamento,
evitando que o modelo aprenda correlações diretas com atributos protegidos.
"""

def suppression(parametros_dataset, dados_dataset_in, parametros_algoritmos, printar=False):
    # Para não alterar os dados originais
    dados_dataset = dados_dataset_in.copy()
    nome_do_teste = dados_dataset['nome_base_de_dados'] + " || SUPPRESSION"

    # Criando uma variável para os dados sensíveis e coluna sensível
    dados_sensiveis = parametros_dataset['dados_sensiveis']
    coluna_sensivel = dados_sensiveis['coluna_sensivel']

    # Excluir a coluna sensível - Próprio da técnica
    X_test_justica = dados_dataset['X_test'].copy()
    dados_dataset['X_train'] = dados_dataset['X_train'].drop(coluna_sensivel, axis=1)
    dados_dataset['X_test'] = dados_dataset['X_test'].drop(coluna_sensivel, axis=1)
    if(printar):
        print(f"A coluna {coluna_sensivel} não foi usada no treinamento do modelo")

    # Concatenando dicionários para usar no treinamento do modelo
    parametros = parametros_dataset | dados_dataset

    # Parâmetros específicos da técnica
    parametros['X_justica'] = True
    parametros['X_test_justica'] = X_test_justica

    desempenho = {}
    for nome_algoritmo in parametros_algoritmos:
        algoritmo = parametros_algoritmos[nome_algoritmo]

        parametros['param_grid'] = algoritmo['parametros_grid']
        parametros['nome_base_de_dados'] = nome_do_teste + f" || {nome_algoritmo.upper().replace('_', ' ')}"
        _, desempenho[nome_algoritmo] = algoritmo['funcao'](**parametros)

    return(desempenho)





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

    df = pd.read_csv(f"Datasets/Processados/Dataset1.csv", sep=',')

    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    parametros_dataset = {}
    parametros_dataset['printar'] = True
    parametros_dataset['preprocessor'] = preprocessor_passthrough
    parametros_dataset['cv_n_splits'] = 2
    parametros_dataset['dados_sensiveis'] = dados_sensiveis_sexo
    
    dados_dataset = {}
    dados_dataset['X_train'] = X_train
    dados_dataset['X_test'] = X_test
    dados_dataset['y_train'] = y_train
    dados_dataset['y_test'] = y_test
    dados_dataset['nome_base_de_dados'] = "DATASET"

    parametros_random_forest = {
        'funcao': random_forest_GSCV,
        'parametros_grid': param_grid_random_forest_basico
    }
    parametros_xgboost = {
        'funcao': xgboost_GSCV,
        'parametros_grid': param_grid_xgboost_basico
    }
    parametros_regressao_logistica = {
        'funcao': regressao_logistica_GSCV,
        'parametros_grid': param_grid_regressao_logistica_basico
    }
    parametros_perceptron = {
        'funcao': perceptron_GSCV,
        'parametros_grid': param_grid_perceptron_basico
    }
    parametros_algoritmos = {
        'xgboost': parametros_xgboost,
        'random_forest': parametros_random_forest,
        'regressao_logistica': parametros_regressao_logistica
    }

    desempenho = suppression(parametros_dataset, dados_dataset, parametros_algoritmos, True)