"""
Este arquivo implementa a técnica de pré-processamento Instance Reweighing (AIF360).
A técnica atribui pesos diferentes às instâncias de treino para mitigar o viés sem alterar os dados.
"""

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
import pandas as pd

# parametros_dataset: preprocessador, dados sensíveis, printar testes, cv_n_splits
# dados_dataset: dados de treino e teste e o nome do caso de teste
# parametros_algoritmos: funções dos algortimos, nomes e respectivos parâmetros de grid

def instance_reweighing(parametros_dataset, dados_dataset_in, parametros_algoritmos, printar=False):
    # Para não alterar os dados originais
    dados_dataset = dados_dataset_in.copy()
    nome_do_teste = dados_dataset['nome_base_de_dados'] + " || INSTANCE REWEIGHING"

    # Definindo os grupos privilegiados e desprivilegiados (lista de dicionário)
    dados_sensiveis = parametros_dataset['dados_sensiveis']
    privileged_groups = [{dados_sensiveis['coluna_sensivel']: dados_sensiveis['grupo_privilegiado']}]
    unprivileged_groups = [{dados_sensiveis['coluna_sensivel']: dados_sensiveis['grupo_desprivilegiado']}]

    # Formando o dataset de treino
    dados_treino = pd.concat([dados_dataset['X_train'], dados_dataset['y_train']], axis=1)

    # Convertendo o DataFrame do Pandas para o formato do AIF360
    dataset_orig = BinaryLabelDataset(
        df=dados_treino,
        favorable_label=dados_sensiveis['rotulo_favoravel'],
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'],
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    dataset_transf = RW.fit_transform(dataset_orig)
    weights = pd.Series(dataset_transf.instance_weights, index=dados_treino.index)

    # Resultados do pré-processamento (pesos encontrados)
    temp_df = dados_treino.copy()
    temp_df['weight'] = weights
    pesos_por_classe = temp_df.groupby([dados_sensiveis['coluna_sensivel'], 'target'])['weight'].mean()
    if(printar):
        print(f"\n----- {nome_do_teste} || PESOS CALCULADOS -----\n")
        print(pesos_por_classe)

    # Concatenando dicionários para usar no treinamento do modelo
    parametros = parametros_dataset | dados_dataset

    # Parâmtros específico da técnica
    parametros['pesos'] = True
    parametros['pesos_modelo'] = weights

    desempenho = {}
    for nome_algoritmo in parametros_algoritmos:
        algoritmo = parametros_algoritmos[nome_algoritmo]

        parametros['param_grid'] = algoritmo['parametros_grid']
        parametros['nome_base_de_dados'] = nome_do_teste + f" || {nome_algoritmo.upper().replace('_', ' ')}"
        _, desempenho[nome_algoritmo] = algoritmo['funcao'](**parametros)

    return(desempenho)





if __name__ == "__main__":
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
        'regressao_logistica': parametros_regressao_logistica,
        'perceptron': parametros_perceptron
    }

    desempenho = instance_reweighing(parametros_dataset, dados_dataset, parametros_algoritmos, True)