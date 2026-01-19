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

def synthetic_data_generation(funcao_modelo, parametros_in, colunas_discretas, printar=False):
    """
    Gera dados sintéticos para equilibrar as taxas de aprovação entre grupos e treina o modelo.
    
    Parâmetros:
    - funcao_modelo: Função de treinamento do modelo.
    - parametros_in: Dicionário com dados e configurações.
    - colunas_discretas: Lista com os nomes das colunas categóricas para o SMOTE-NC.
    - printar: Booleano para exibir estatísticas das razões de aprovação.
    
    Retorna:
    - desempenho: Dicionário com os resultados da avaliação.
    """
    parametros = parametros_in.copy()
    dados_sensiveis = parametros['dados_sensiveis']

    # Ajustando o nome
    nome_banco_treino = parametros['nome_base_de_dados'] + " || DADOS DE TREINO || SYNTHETIC DATA GENERATION"

    coluna_sensivel = dados_sensiveis['coluna_sensivel']

    # Formando o dataset de treino
    dados_treino = pd.concat([parametros['X_train'], parametros['y_train']], axis=1)

    # Descobrindo a razão para o target de cada sexo
    contagem_grupos = dados_treino.groupby([coluna_sensivel, 'target']).size()
    razao_target_mulheres = contagem_grupos.loc[(0, 1)] / contagem_grupos.loc[(0, 0)]
    razao_target_homens = contagem_grupos.loc[(1, 1)] / contagem_grupos.loc[(1, 0)]

    if(printar):
        print(f"\n----- {nome_banco_treino} || DISTRIBUIÇÃO DOS DADOS DE TREINO RECEBIDOS\n")
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
    nome_banco_sintetico = parametros['nome_base_de_dados'] + " || DADOS DE TREINO COM SDG || SYNTHETIC DATA GENERATION"

    contagem_grupos = df_synthetic_data_generation.groupby([coluna_sensivel, 'target']).size()
    razao_target_mulheres = contagem_grupos.loc[(0, 1)] / contagem_grupos.loc[(0, 0)]
    razao_target_homens = contagem_grupos.loc[(1, 1)] / contagem_grupos.loc[(1, 0)]

    if(printar):
        print(f"Tamanho total do DataFrame de treino: {dados_treino.shape[0]} linhas")
        print(f"\n----- {nome_banco_sintetico} || DISTRIBUIÇÃO FINAL DOS DADOS DE TREINO\n")
        print(df_synthetic_data_generation.groupby([coluna_sensivel, 'target']).size())
        print("Razão de target = 1 / target = 0: ")
        print(f"Mulheres: {razao_target_mulheres}")
        print(f"Homens: {razao_target_homens}")
        print(f"Tamanho total do novo DataFrame de treino: {df_synthetic_data_generation.shape[0]} linhas")

    # Balanceando os dados novos
    base_de_dados_final = df_synthetic_data_generation
    nome_final = nome_banco_sintetico

    if("BANCEADO" in parametros['nome_base_de_dados']):
        base_de_dados_final, estatisticas = balancear(df_synthetic_data_generation, nome_banco_sintetico, 'target', printar=printar)
        nome_final = parametros['nome_base_de_dados'] + " COM SDG BALANCEADO || SYNTHETIC DATA GENERATION"

    parametros['nome_base_de_dados'] = nome_final
    
    # Atualizando os dados de treino para o modelo
    parametros['X_train'] = base_de_dados_final.drop('target', axis=1)
    parametros['y_train'] = base_de_dados_final['target']

    # TREINANDO UM MODELO

    # Parâmetros já contém os dados de treino e teste e o nome dos dados
    # Parâmetros específicos dessa técnica - Não há

    _, desempenho = funcao_modelo(**parametros)

    return(desempenho)

if __name__ == "__main__":
    from c0_1_configuracoes import (
        param_grid_perceptron_basico,
        preprocessor_passthrough,
        dados_sensiveis_age,
        colunas_discretas2
        )
    from c2_4_perceptron import perceptron_GSCV
    from c1_3_preprocessor_automatizado import aplicar_pre_processador

    df = pd.read_csv(f"Datasets/Processados/Dataset2.csv", sep=',')

    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    parametros= {}
    parametros['printar'] = True
    parametros['matriz_de_confusao'] = False
    parametros['param_grid'] = param_grid_perceptron_basico
    parametros['preprocessor'] = preprocessor_passthrough
    parametros['cv_n_splits'] = 2
    parametros['X_train'] = X_train
    parametros['X_test'] = X_test
    parametros['y_train'] = y_train
    parametros['y_test'] = y_test
    parametros['nome_base_de_dados'] = "DATASET"
    parametros['dados_sensiveis'] = dados_sensiveis_age

    desempenho = synthetic_data_generation(perceptron_GSCV, parametros, colunas_discretas2, printar=True)