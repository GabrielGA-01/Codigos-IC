def suppression(funcao_modelo, parametros_in, printar=False):
    parametros = parametros_in.copy()
    dados_sensiveis = parametros['dados_sensiveis']

    # Ajustando o nome
    parametros['nome_base_de_dados'] = parametros['nome_base_de_dados'] + " || SUPPRESSION"

    coluna_sensivel = dados_sensiveis['coluna_sensivel']

    # TREINANDO UM MODELO

    # Excluir a coluna sensível - Próprio da técnica
    X_test_justica = parametros['X_test'].copy()
    parametros['X_train'] = parametros['X_train'].drop(coluna_sensivel, axis=1)
    parametros['X_test'] = parametros['X_test'].drop(coluna_sensivel, axis=1)
    if(printar):
        print(f"A coluna {coluna_sensivel} não foi usada no treinamento do modelo")

    # Parâmetros já contém os dados de treino e teste e o nome dos dados
    # Parâmetros específicos dessa técnica
    parametros['X_justica'] = True
    parametros['X_test_justica'] = X_test_justica

    _, desempenho = funcao_modelo(**parametros)

    return(desempenho)

if __name__ == "__main__":
    import pandas as pd

    from c0_1_configuracoes import (
        param_grid_regressao_logistica_basico,
        preprocessor_passthrough,
        dados_sensiveis_sexo
        )
    from c2_3_regressao_logistica import regressao_logistica_GSCV
    from c1_3_preprocessor_automatizado import aplicar_pre_processador

    df = pd.read_csv(f"Datasets/Processados/Dataset1.csv", sep=',')

    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    parametros= {}
    parametros['printar'] = True
    parametros['matriz_de_confusao'] = False
    parametros['param_grid'] = param_grid_regressao_logistica_basico
    parametros['preprocessor'] = preprocessor_passthrough
    parametros['cv_n_splits'] = 2
    parametros['X_train'] = X_train
    parametros['X_test'] = X_test
    parametros['y_train'] = y_train
    parametros['y_test'] = y_test
    parametros['nome_base_de_dados'] = "DATASET"
    parametros['dados_sensiveis'] = dados_sensiveis_sexo

    suppression(regressao_logistica_GSCV, parametros, printar=True)