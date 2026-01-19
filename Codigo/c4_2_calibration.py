"""
Este arquivo implementa uma técnica de calibração de probabilidades por grupo.
Utiliza a Isotonic Regression para calibrar as probabilidades de saída do modelo separadamente
para cada grupo sensível, buscando que as probabilidades reflitam fielmente as frequências reais.
"""

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import pandas as pd

from c0_1_configuracoes import seed
from c1_4_desempenho_modelos import desempenho_modelo

def calibration(funcao_modelo, parametros_in, printar=False):
    """
    Treina o modelo e aplica calibração isotônica separada por grupo (ex: Homens e Mulheres).
    
    Parâmetros:
    - funcao_modelo: Função de treinamento do modelo base.
    - parametros_in: Dicionário com dados e configurações.
    - printar: Booleano para exibir os resultados do modelo calibrado.
    
    Retorna:
    - desempenho_base: Resultados do modelo original.
    - desempenho_calibration: Resultados do modelo após calibração por grupo.
    """
    parametros = parametros_in.copy()
    dados_sensiveis = parametros['dados_sensiveis']

    # Ajustando o nome
    parametros['nome_base_de_dados'] = parametros['nome_base_de_dados'] + " || CALIBRATION"

    coluna_sensivel = dados_sensiveis['coluna_sensivel']

    # Separando os dados de treino (70%*) em treino (55%*) e calibração (15%*) *Em relação ao total de dados
    X_train, X_calib, y_train, y_calib = train_test_split(parametros['X_train'], parametros['y_train'], test_size=15/70, random_state=seed, stratify=parametros['y_train'])

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

    # CALIBRATION

    X_test = parametros['X_test']
    y_test = parametros['y_test']

    # Obter a probabilidade da classe positiva (TARGET=1)
    probs_calib = model.predict_proba(X_calib)[:, 1]

    # Isolar dados de calibração para homens e mulheres
    is_male_calib = (X_calib[coluna_sensivel] == 1)
    is_female_calib = (X_calib[coluna_sensivel] == 0)

    # Calibrador para o grupo de homens
    calibrator_male = IsotonicRegression(out_of_bounds='clip')
    calibrator_male.fit(probs_calib[is_male_calib], y_calib[is_male_calib])

    # Calibrador para o grupo de mulheres
    calibrator_female = IsotonicRegression(out_of_bounds='clip')
    calibrator_female.fit(probs_calib[is_female_calib], y_calib[is_female_calib])

    # Obter probabilidades do modelo principal no conjunto de teste
    probs_test = model.predict_proba(X_test)[:, 1]

    # Separar os dados de teste por grupo
    is_male_test = (X_test[coluna_sensivel] == 1)
    is_female_test = (X_test[coluna_sensivel] == 0)

    # Aplicar o calibrador correspondente a cada grupo
    calibrated_probs_test = pd.Series(index=X_test.index, dtype=float)
    calibrated_probs_test[is_male_test] = calibrator_male.predict(probs_test[is_male_test])
    calibrated_probs_test[is_female_test] = calibrator_female.predict(probs_test[is_female_test])

    # Resultados
    y_pred_calibration = (calibrated_probs_test >= 0.5).astype(int)
    desempenho_calibration = desempenho_modelo(parametros['nome_base_de_dados'], model, parametros['nome_modelo'], X_train, X_test, X_test, y_pred_calibration, y_test, dados_sensiveis, printar=printar_resultado, matriz_de_confusao=matriz_de_confusao_resultado)

    return(desempenho_base, desempenho_calibration)

if __name__ == "__main__":
    from c0_1_configuracoes import (
        param_grid_xgboost_basico,
        preprocessor_passthrough,
        dados_sensiveis_age
        )
    from c2_2_xgboost import xgboost_GSCV
    from c1_3_preprocessor_automatizado import aplicar_pre_processador

    df = pd.read_csv(f"Datasets/Processados/Dataset3.csv", sep=',')

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

    desempenho_orig, desempenho = calibration(xgboost_GSCV, parametros, printar=True)