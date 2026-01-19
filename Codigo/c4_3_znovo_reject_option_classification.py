"""
Este arquivo implementa a técnica de pós-processamento Reject Option Classification (AIF360).
Esta técnica altera previsões feitas em uma área de incerteza (próximo ao threshold) 
favorecendo o grupo desprivilegiado para melhorar a justiça do modelo.
"""

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import RejectOptionClassification
from sklearn.model_selection import train_test_split

from c0_1_configuracoes import seed
from c1_4_desempenho_modelos import desempenho_modelo

def reject_option_classification(parametros_dataset_pos, dados_dataset, modelo, nome_modelo):
    # Para não alterar os dados originais (os dados de treino, validação e teste não são alterados aqui)
    nome_do_teste = dados_dataset['nome_base_de_dados'] + f" || REJECT OPTION CLASSIFICATION || {nome_modelo.upper().replace('_', ' ')}"

    # Criando uma variável para os dados sensíveis
    dados_sensiveis = parametros_dataset_pos['dados_sensiveis']

    X_train = dados_dataset['X_train']

    X_val = dados_dataset['X_val']
    y_val = dados_dataset['y_val']

    X_test = dados_dataset['X_test']
    y_test = dados_dataset['y_test']

    # Determinando a probabilidade das decisões para o conjunto de validação e tornando binário (>50% (1) ou <50% (0))
    y_pred_proba_val = modelo.predict_proba(X_val)[:, dados_sensiveis['rotulo_favoravel']]
    y_pred_base_val = modelo.predict(X_val)

    # Juntando as features e rótulos de validação real e previstos
    df_true_val = X_val.copy()
    df_true_val['target'] = y_val
    dataset_true_val = BinaryLabelDataset(
        df=df_true_val, 
        favorable_label=dados_sensiveis['rotulo_favoravel'], 
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'], 
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )

    df_pred_val = X_val.copy()
    df_pred_val['target'] = y_pred_base_val
    dataset_pred_val = BinaryLabelDataset(
        df=df_pred_val, 
        favorable_label=dados_sensiveis['rotulo_favoravel'], 
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'], 
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )
    dataset_pred_val.scores = y_pred_proba_val.reshape(-1, 1)

    # Aplicando o Reject Option Classification
    privileged_groups = [{dados_sensiveis['coluna_sensivel']: dados_sensiveis['grupo_privilegiado']}]
    unprivileged_groups = [{dados_sensiveis['coluna_sensivel']: dados_sensiveis['grupo_desprivilegiado']}]

    # Margem de incerteza em que pode mudar o rótulo previsto
    low_class_thresh = 0.45
    high_class_thresh = 0.55

    roc = RejectOptionClassification(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        low_class_thresh=low_class_thresh,
        high_class_thresh=high_class_thresh
    )

    # Ajuste o algoritmo para aprender as regras
    roc = roc.fit(dataset_true_val, dataset_pred_val)

    # Determinando a previsão nos dados de teste
    y_pred_proba_test = modelo.predict_proba(X_test)[:, dados_sensiveis['rotulo_favoravel']]
    y_pred_base_test = modelo.predict(X_test)

    df_pred_test = X_test.copy()
    df_pred_test['target'] = y_pred_base_test
    dataset_pred_test = BinaryLabelDataset(
        df=df_pred_test, 
        favorable_label=dados_sensiveis['rotulo_favoravel'], 
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'],
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )
    dataset_pred_test.scores = y_pred_proba_test.reshape(-1, 1)

    # .predict() para obter o dataset com os rótulos corrigidos
    dataset_transf = roc.predict(dataset_pred_test)

    # Resultados
    y_pred_reject_option_classification = dataset_transf.labels.ravel()
    desempenho_reject_option_classification = desempenho_modelo(nome_do_teste, modelo, nome_modelo, X_train, X_test, X_test, y_pred_reject_option_classification, y_test, dados_sensiveis, printar=parametros_dataset_pos['printar'])

    return(desempenho_reject_option_classification)






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

    desempenho = reject_option_classification(parametros_dataset_pos, dados_dataset, modelo, nome_modelo)