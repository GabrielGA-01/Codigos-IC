"""
Este arquivo implementa uma técnica customizada de balanceamento sintético para justiça.
Ele utiliza o SMOTENC para equilibrar a proporção de rótulos favoráveis e desfavoráveis 
dentro de cada grupo de uma variável sensível.
"""

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn import config_context 

from c0_1_configuracoes import seed
from c1_1_balancear import balancear_proporcao_rotulo_binario as balancear

def synthetic_data_generation(parametros_dataset, dados_dataset_in, parametros_algoritmos, colunas_discretas, printar=False):
    # Para não alterar os dados originais
    dados_dataset = dados_dataset_in.copy()
    nome_do_treino = dados_dataset['nome_base_de_dados'] + " || DADOS DE TREINO || SYNTHETIC DATA GENERATION"

    # Criando uma variável para os dados sensíveis e coluna sensível
    dados_sensiveis = parametros_dataset['dados_sensiveis']
    coluna_sensivel = dados_sensiveis['coluna_sensivel']

    # Formando o dataset de treino
    dados_treino = pd.concat([dados_dataset['X_train'], dados_dataset['y_train']], axis=1)

    # Descobrindo a razão para o target de cada sexo
    contagem_grupos = dados_treino.groupby([coluna_sensivel, 'target']).size()
    razao_target_mulheres = contagem_grupos.loc[(0, 1)] / contagem_grupos.loc[(0, 0)]
    razao_target_homens = contagem_grupos.loc[(1, 1)] / contagem_grupos.loc[(1, 0)]

    if(printar):
        print(f"\n----- {nome_do_treino} || DISTRIBUIÇÃO DOS DADOS DE TREINO RECEBIDOS\n")
        print(contagem_grupos)
        print("Razão de target == 1 / target == 0: ")
        print(f"Mulheres: {razao_target_mulheres}")
        print(f"Homens: {razao_target_homens}\n")

    # Separar os dados por gênero
    df_mulheres = dados_treino[dados_treino[coluna_sensivel] == 0].copy()
    df_homens = dados_treino[dados_treino[coluna_sensivel] == 1].copy()

    # Separar features (X) e target (y) para o subgrupo de mulheres
    X_mulheres = df_mulheres.drop('target', axis=1)
    y_mulheres = df_mulheres['target']

    # Identificando as colunas categóricas
    categorical_features_indices = [X_mulheres.columns.get_loc(col) for col in colunas_discretas]

    if(razao_target_mulheres > razao_target_homens):
        nova_quantidade = int(contagem_grupos.loc[(0, 1)] / razao_target_homens)
        estrategia_mulheres = {0: nova_quantidade}
        if(printar):
            print(f"Aumentando o número de mulheres com target == 0 para {nova_quantidade}")
    else:
        nova_quantidade = int(contagem_grupos.loc[(0, 0)] * razao_target_homens)
        estrategia_mulheres = {1: nova_quantidade}
        if(printar):
            print(f"Aumentando o número de mulheres com target == 1 para {nova_quantidade}")

    smotenc = SMOTENC(sampling_strategy=estrategia_mulheres, categorical_features=categorical_features_indices, random_state=seed)

    X_mulheres_np = X_mulheres.to_numpy()
    y_mulheres_np = y_mulheres.to_numpy()

    # Aplicar o SMOTENC usando o retorno no formato np
    with config_context(transform_output="default"):
        X_mulheres_resampled_np, y_mulheres_resampled_np = smotenc.fit_resample(X_mulheres_np, y_mulheres_np)

    # Reconstruir o DataFrame de mulheres com os novos dados sintéticos
    df_mulheres_resampled = pd.concat([
        pd.DataFrame(X_mulheres_resampled_np, columns=X_mulheres.columns),
        pd.Series(y_mulheres_resampled_np, name='target')
    ], axis=1)

    # Juntar o grupo de homens (intocado) com o novo grupo de mulheres (modificado)
    df_synthetic_data_generation = pd.concat([df_homens, df_mulheres_resampled], ignore_index=True)
    nome_do_treino_sintetico = dados_dataset['nome_base_de_dados'] + " || DADOS DE TREINO COM SDG || SYNTHETIC DATA GENERATION"

    contagem_grupos = df_synthetic_data_generation.groupby([coluna_sensivel, 'target']).size()
    razao_target_mulheres = contagem_grupos.loc[(0, 1)] / contagem_grupos.loc[(0, 0)]
    razao_target_homens = contagem_grupos.loc[(1, 1)] / contagem_grupos.loc[(1, 0)]

    if(printar):
        print(f"Tamanho total do DataFrame de treino: {dados_treino.shape[0]} linhas")
        print(f"\n----- {nome_do_treino_sintetico} || DISTRIBUIÇÃO FINAL DOS DADOS DE TREINO\n")
        print(df_synthetic_data_generation.groupby([coluna_sensivel, 'target']).size())
        print("Razão de target = 1 / target = 0: ")
        print(f"Mulheres: {razao_target_mulheres}")
        print(f"Homens: {razao_target_homens}")
        print(f"Tamanho total do novo DataFrame de treino: {df_synthetic_data_generation.shape[0]} linhas")

    # Balanceando os dados novos
    base_de_dados_final = df_synthetic_data_generation
    nome_do_teste = nome_do_treino_sintetico

    if("BALANCEADO" in dados_dataset['nome_base_de_dados']):
        base_de_dados_final, estatisticas = balancear(df_synthetic_data_generation, nome_do_treino_sintetico, 'target', printar=printar)
        nome_do_teste = dados_dataset['nome_base_de_dados'] + " COM SDG BALANCEADO || SYNTHETIC DATA GENERATION"
    
    # Atualizando os dados de treino para o modelo
    dados_dataset['X_train'] = base_de_dados_final.drop('target', axis=1)
    dados_dataset['y_train'] = base_de_dados_final['target']

    # Concatenando dicionários para usar no treinamento do modelo
    parametros = parametros_dataset | dados_dataset

    # Parâmtros específico da técnica - Não há

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
        colunas_discretas1
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
    parametros_dataset['dados_sensiveis'] = dados_sensiveis_age
    
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

    desempenho = synthetic_data_generation(parametros_dataset, dados_dataset, parametros_algoritmos, colunas_discretas1, True)