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

def reject_option_classification(funcao_modelo, parametros_in, printar=False):
    """
    Treina o modelo e aplica o Reject Option Classification para ajustar previsões incertas.
    
    Parâmetros:
    - funcao_modelo: Função de treinamento do modelo base.
    - parametros_in: Dicionário com dados e configurações.
    - printar: Booleano para exibir os resultados do modelo pós-processado.
    
    Retorna:
    - desempenho_base: Resultados do modelo original.
    - desempenho_reject_option_classification: Resultados do modelo após aplicação da técnica.
    """
    parametros = parametros_in.copy()
    dados_sensiveis = parametros['dados_sensiveis']

    # Ajustando o nome
    parametros['nome_base_de_dados'] = parametros['nome_base_de_dados'] + " || REJECT OPTION CLASSIFICATION"

    # Separando os dados de treino (70%*) em treino (55%*) e calibração (15%*) *Em relação ao total de dados
    X_train, X_val, y_train, y_val = train_test_split(parametros['X_train'], parametros['y_train'], test_size=15/70, random_state=seed, stratify=parametros['y_train'])

    parametros['X_train'] = X_train
    parametros['y_train'] = y_train

    # TREINANDO UM MODELO

    # Parâmetros já contém os dados de treino e teste e o nome dos dados

    # Corrigindo o print e a matriz de confusão
    if 'printar' in parametros.keys():
        printar_resultado = parametros['printar']
        parametros['printar'] = False # Não printar o desempenho modelo original
    else:
        printar_resultado = False

    if 'matriz_de_confusao' in parametros.keys():
        matriz_de_confusao_resultado = parametros['matriz_de_confusao']
        parametros['matriz_de_confusao'] = False # Não printar o desempenho modelo original
    else:
        matriz_de_confusao_resultado = False

    # Treinar o modelo
    model, desempenho_base = funcao_modelo(**parametros)

    # REJECT OPTION CLASSIFICATION

    X_test = parametros['X_test']
    y_test = parametros['y_test']

    # Determinando a probabilidade das decisões para o conjunto de validação e tornando binário (>50% (1) ou <50% (0))
    y_pred_proba_val = model.predict_proba(X_val)[:, dados_sensiveis['rotulo_favoravel']]
    y_pred_base_val = model.predict(X_val)

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
    y_pred_proba_test = model.predict_proba(X_test)[:, dados_sensiveis['rotulo_favoravel']]
    y_pred_base_test = model.predict(X_test)

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
    desempenho_reject_option_classification = desempenho_modelo(parametros['nome_base_de_dados'], model, parametros['nome_modelo'], X_train, X_test, X_test, y_pred_reject_option_classification, y_test, dados_sensiveis, printar=printar_resultado, matriz_de_confusao=matriz_de_confusao_resultado)

    return(desempenho_base, desempenho_reject_option_classification)

if __name__ == "__main__":
    import pandas as pd

    from c0_1_configuracoes import (
        param_grid_xgboost_basico,
        preprocessor_passthrough,
        dados_sensiveis_age
        )
    from c2_2_xgboost import xgboost_GSCV
    from c1_3_preprocessor_automatizado import aplicar_pre_processador

    df = pd.read_csv(f"Datasets/Processados/Dataset1.csv", sep=',')

    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    parametros= {}
    parametros['printar'] = True
    parametros['matriz_de_confusao'] = False
    parametros['param_grid'] = param_grid_xgboost_basico
    parametros['preprocessor'] = preprocessor_passthrough
    parametros['cv_n_splits'] = 2
    parametros['X_train'] = X_train
    parametros['X_test'] = X_test
    parametros['y_train'] = y_train
    parametros['y_test'] = y_test
    parametros['nome_base_de_dados'] = "DATASET"
    parametros['dados_sensiveis'] = dados_sensiveis_age

    desempenho_orig, desempenho = reject_option_classification(xgboost_GSCV, parametros, printar=True)