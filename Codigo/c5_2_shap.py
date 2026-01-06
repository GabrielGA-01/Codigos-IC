import shap
import numpy as np
import pandas as pd

def importancia_shap(pipeline, nome_modelo, nome_dataset, cenario, X_test):
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    X_test_transf = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Conversão para dataframe
    if not isinstance(X_test_transf, pd.DataFrame):
        X_test_transf = pd.DataFrame(X_test_transf, columns=feature_names)

    if nome_modelo.lower().replace(" ", "_") in ['xgboost', 'random_forest']:
        explainer = shap.TreeExplainer(classifier)
    elif nome_modelo.lower().replace(" ", "_") in ['regressao_logistica', 'perceptron']:
        explainer = shap.LinearExplainer(classifier, X_test_transf)
    else:
        raise ValueError(f"Modelo '{nome_modelo}' não suportado pelo SHAP.")
    
    shap_values = explainer.shap_values(X_test_transf)

    # Padronizando o resultado

    # Caso o retorno seja uma lista com dois conjuntos de dados [importância_classe1, importância_classe2]
    # Obtém-se a média dos valores absolutos de cada variável
    if isinstance(shap_values, list):
        shap_values_abs = np.abs(shap_values[1]).mean(axis=0)
    # Caso seja um array
    # Verifica se ele é multiclasse ou simples.
    else:
        if len(shap_values.shape) == 3:
             shap_values_abs = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
             shap_values_abs = np.abs(shap_values).mean(axis=0)

    return [
        {
            "dataset": nome_dataset,
            "cenario": cenario,
            "modelo": nome_modelo,
            "variavel": variavel,
            "importancia": importance
        }
        for variavel, importance in zip(feature_names, shap_values_abs)
    ]
