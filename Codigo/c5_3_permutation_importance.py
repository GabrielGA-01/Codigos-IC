"""
Este arquivo implementa o cálculo da importância das variáveis via Permutação.
A técnica consiste em embaralhar os valores de cada coluna e observar a queda no desempenho do modelo.
"""

from sklearn.inspection import permutation_importance
import pandas as pd 

from c0_1_configuracoes import seed

def importancia_permutacao(pipeline, nome_modelo, nome_dataset, cenario, X_test, y_test):
    """
    Calcula a importância das variáveis via permutação.
    
    Parâmetros:
        - pipeline: pipeline treinado (com preprocessor + modelo)
        - modelo: nome do modelo (str)
        - arquivo: nome do dataset (str)
        - label: cenário (str)
        - X_test: base de validação original (sem pré-processar)
        - y_test: target de validação
    Retorna:
        - Lista de dicionários com importâncias
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Aplicar o pré-processamento no conjunto de validação
    X_test_transf = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Garantir que seja um DataFrame
    if not isinstance(X_test_transf, pd.DataFrame):
        X_test_transf = pd.DataFrame(X_test_transf, columns=feature_names)

    # Importância por permutação
    result = permutation_importance(
        classifier,
        X_test_transf,
        y_test,
        n_repeats=10,
        random_state=seed,
        scoring='f1_macro'
    )

    return [
        {
            "dataset": nome_dataset,
            "cenario": cenario,
            "modelo": nome_modelo,
            "variavel": name,
            "importancia": importance
        }
        for name, importance in zip(feature_names, result.importances_mean)
    ]