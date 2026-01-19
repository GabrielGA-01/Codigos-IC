"""
Este arquivo implementa o cálculo da importância das variáveis utilizando SHAP (SHapley Additive exPlanations).
Ele seleciona automaticamente o melhor Explainer (Tree, Linear ou Kernel) dependendo do modelo fornecido.
"""

import shap
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

def importancia_shap(pipeline, nome_modelo, nome_dataset, cenario, X_test):
    """
    Calcula os valores SHAP médios para cada feature do modelo.
    
    Parâmetros:
    - pipeline: Pipeline treinado contendo preprocessor e classifier.
    - nome_modelo: String com o nome do modelo para decidir o Explainer.
    - nome_dataset: String com o nome do dataset.
    - cenario: String descrevendo o cenário de teste.
    - X_test: DataFrame de teste (original, será transformado internamente).
    
    Retorna:
    - Lista de dicionários contendo a importância (média do valor absoluto) de cada variável.
    """
    # Preparação dos dados
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    X_test_transf = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Conversão garantida para DataFrame para manter consistência
    if not isinstance(X_test_transf, pd.DataFrame):
        X_test_transf = pd.DataFrame(X_test_transf, columns=feature_names)

    # O SHAP transforma os dados em NumPy. Esta função força a volta para DataFrame.
    def predict_fn_wrapper(X):
        X_df = pd.DataFrame(X, columns=feature_names)
        if hasattr(classifier, 'predict_proba'):
            return classifier.predict_proba(X_df)
        elif hasattr(classifier, 'decision_function'):
            return classifier.decision_function(X_df)
        else:
            return classifier.predict(X_df)

    # Identificação do Explainer adequado
    explainer = None
    shap_values = None

    # Caso A: CalibratedClassifierCV ou Modelos Desconhecidos (Usa KernelExplainer)
    if isinstance(classifier, CalibratedClassifierCV) or nome_modelo.lower().replace(" ", "_") not in ['xgboost', 'random_forest', 'regressão_logística', 'perceptron']:
        print(f"Usando KernelExplainer para {nome_modelo}...")
        
        # O KernelExplainer é lento. Usamos kmeans para resumir o background (10 amostras)
        background = shap.kmeans(X_test_transf, 10) if len(X_test_transf) > 10 else X_test_transf
        
        # Passamos a nossa função wrapper em vez do predict_proba direto
        explainer = shap.KernelExplainer(predict_fn_wrapper, background)
        shap_values = explainer.shap_values(X_test_transf)

    # Caso B: Modelos de Árvore (TreeExplainer é muito mais rápido)
    elif nome_modelo.lower().replace(" ", "_") in ['xgboost', 'random_forest']:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_transf)
        
    # Caso C: Modelos Lineares Puros (LinearExplainer)
    elif nome_modelo.lower().replace(" ", "_") in ['regressão_logística', 'perceptron']:
        try:
            # LinearExplainer geralmente lida bem com nomes se passarmos o X_test_transf
            explainer = shap.LinearExplainer(classifier, X_test_transf)
            shap_values = explainer.shap_values(X_test_transf)
        except Exception as e:
            print(f"LinearExplainer falhou ({e}). Tentando KernelExplainer com wrapper...")
            background = shap.kmeans(X_test_transf, 10) if len(X_test_transf) > 10 else X_test_transf
            explainer = shap.KernelExplainer(predict_fn_wrapper, background)
            shap_values = explainer.shap_values(X_test_transf)

    # Padronização do Resultado (Tratamento de dimensões)
    # O SHAP pode retornar:
    # - Uma lista de arrays (ex: [shap_classe_0, shap_classe_1])
    # - Um array 3D (amostras, features, classes)
    # - Um array 2D (amostras, features)
    
    if isinstance(shap_values, list):
        # Para classificação binária, é usada a importância da classe positiva (índice 1)
        # Se for multiclasse, aqui pegaria a classe 1 (ajuste se necessário)
        shap_values_abs = np.abs(shap_values[1]).mean(axis=0)
    else:
        if len(shap_values.shape) == 3:
             # Caso array 3D: (n_samples, n_features, n_classes)
             shap_values_abs = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
             # Caso array 2D: (n_samples, n_features)
             shap_values_abs = np.abs(shap_values).mean(axis=0)

    # Formatação Final
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