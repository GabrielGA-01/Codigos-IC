"""
Este arquivo implementa o cálculo da importância das variáveis utilizando LIME.
Ele gera explicações locais para uma amostra de instâncias e as agrega para obter uma visão global.
"""

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

def importancia_lime(pipeline, nome_modelo, nome_dataset, cenario, X_train, X_test, n_samples=100):
    """
    Calcula a importância LIME agregada para as features do modelo.
    
    Parâmetros:
    - pipeline: Pipeline treinado.
    - nome_modelo: Nome do modelo.
    - nome_dataset: Nome do dataset.
    - cenario: Cenário de teste.
    - X_train: Dados de treino processados (necessário para o explainer LIME).
    - X_test: Dados de teste.
    - n_samples: Número de amostras locais a serem explicadas e agregadas.
    
    Retorna:
    - Lista de dicionários com a importância média absoluta de cada variável.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    X_train_transf = preprocessor.transform(X_train)
    X_test_transf = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Garantir que sejam DataFrames
    if not isinstance(X_train_transf, pd.DataFrame):
        X_train_transf = pd.DataFrame(X_train_transf, columns=feature_names)
    
    if not isinstance(X_test_transf, pd.DataFrame):
         X_test_transf = pd.DataFrame(X_test_transf, columns=feature_names)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_transf.values,
        feature_names=feature_names,
        class_names=["Classe_0", "Classe_1"],
        mode="classification"
    )

    # Garantir que n_samples não seja maior que o tamanho do teste
    n_samples_real = min(n_samples, len(X_test_transf))
    indices = np.random.choice(range(len(X_test_transf)), size=n_samples_real, replace=False)
    importancias_acumuladas = {name: 0.0 for name in feature_names}

    # Função wrapper para garantir que o modelo receba DataFrame com nomes de colunas
    def predict_wrapper(data_array):
        df_temp = pd.DataFrame(data_array, columns=feature_names)
        return classifier.predict_proba(df_temp)

    for i in indices:
        exp = explainer.explain_instance(
            data_row=X_test_transf.iloc[i].values,
            predict_fn=predict_wrapper
        )
        
        # exp.local_exp[1] contém uma lista de tuplas: (id_da_variavel, peso)
        for feature_id, weight in exp.local_exp[1]:
            feature_name = feature_names[feature_id]
            importancias_acumuladas[feature_name] += abs(weight)

    return [
        {
            "dataset": nome_dataset,
            "cenario": cenario,
            "modelo": nome_modelo,
            "variavel": name,
            "importancia": importancias_acumuladas[name] / len(indices)
        }
        for name in feature_names

    ]