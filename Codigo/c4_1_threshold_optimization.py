from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split

from c0_configuracoes import seed
from c1_4_desempenho_modelos import desempenho_modelo


def threshold_optimization(funcao_modelo, parametros_in, printar=False):
    parametros = parametros_in.copy()
    dados_sensiveis = parametros['dados_sensiveis']

    # Ajustando o nome
    parametros['nome_base_de_dados'] = parametros['nome_base_de_dados'] + " || THRESHOLD OPTIMIZATION"

    coluna_sensivel = dados_sensiveis['coluna_sensivel']

    # Separando os dados de treino (70%*) em treino (55%*) e validação (15%*) *Em relação ao total de dados
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

    # THRESHOLD OPTIMIZATION

    X_test = parametros['X_test']
    y_test = parametros['y_test']

    # Atributo pessoal
    A_val = X_val[coluna_sensivel]
    A_test = X_test[coluna_sensivel]

    optimizer = ThresholdOptimizer(estimator=model, constraints='demographic_parity', prefit=True)
    optimizer.fit(X_val, y_val, sensitive_features=A_val)
    y_pred_threshold_optimization = optimizer.predict(X_test, sensitive_features=A_test)

    desempenho_threshold_optimization = desempenho_modelo(parametros['nome_base_de_dados'], model, X_test, X_test, y_pred_threshold_optimization, y_test, dados_sensiveis, printar=printar_resultado, matriz_de_confusao=matriz_de_confusao_resultado)

    return(desempenho_base, desempenho_threshold_optimization)

if __name__ == "__main__":
    import pandas as pd

    from c0_configuracoes import (
        param_grid_xgboost_basico,
        preprocessor_passthrough,
        dados_sensiveis_age
        )
    from c2_2_xgboost import xgboost_GSCV
    from c1_3_preprocessor_automatizado import aplicar_pre_processador

    df = pd.read_csv(f"Datasets/Processados/Dataset2.csv", sep=',')

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

    desempenho_orig, desempenho = threshold_optimization(xgboost_GSCV, parametros, printar=True)