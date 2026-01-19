"""
Este arquivo contém a implementação do treinamento do modelo Perceptron utilizando 
Pipeline, GridSearchCV, StratifiedKFold e CalibratedClassifierCV para obtenção de probabilidades.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV


from c0_1_configuracoes import seed
from c1_4_desempenho_modelos import desempenho_modelo

def perceptron_GSCV(param_grid, preprocessor, cv_n_splits, nome_base_de_dados, X_train, X_test, y_train, y_test, dados_sensiveis, printar=False, matriz_de_confusao=False, grafico_shap=False, pesos=False, pesos_modelo=None, X_justica=False, X_test_justica=None, nome_modelo='perceptron'):
  """
  Executa a busca de hiperparâmetros e treina o Perceptron, avaliando seu desempenho e os coeficientes das features.
  
  Parâmetros:
  - param_grid: Dicionário com a grade de parâmetros para o GridSearchCV.
  - preprocessor: Objeto ColumnTransformer para pré-processamento dos dados.
  - cv_n_splits: Número de divisões para a validação cruzada.
  - nome_base_de_dados: String com o nome do dataset.
  - X_train, X_test: DataFrames de treino e teste.
  - y_train, y_test: Séries com os rótulos de treino e teste.
  - dados_sensiveis: Dicionário com config da variável sensível para métricas de justiça.
  - printar: Booleano para exibir resultados no console.
  - matriz_de_confusao: Booleano para gerar gráfico da matriz de confusão.
  - grafico_shap: Booleano para gerar gráfico (não implementado nativamente para Perceptron neste script).
  - pesos: Booleano indicando se deve usar pesos de amostra (sample_weight).
  - pesos_modelo: Séries com os pesos para cada instância de treino.
  - X_justica: Booleano para usar um dataset de teste diferente para justiça.
  - X_test_justica: Dataset de teste original para métricas de justiça.
  - nome_modelo: Identificador string do modelo.
  
  Retorna:
  - best_model: O melhor modelo (pipeline) encontrado pelo GridSearch.
  - desempenho: Dicionário com as métricas de avaliação e parâmetros otimizados.
  """
  desempenho = {}

  # Instanciando o modelo
  base_model = Perceptron()

  # Para obter a probabilidade da previsão
  calibrated_model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')

  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', calibrated_model)
  ])

  cv_splitter = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=seed)
  grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1_macro', cv=cv_splitter, n_jobs=-1, verbose=0)

  fit_params = {}
  if pesos and pesos_modelo is not None:
      pesos_treino = pesos_modelo.loc[X_train.index]
      fit_params['classifier__sample_weight'] = pesos_treino

  grid_search.fit(X_train, y_train, **fit_params)

  best_model = grid_search.best_estimator_
  y_pred = best_model.predict(X_test)

  if(X_justica == False):
    X_test_justica = X_test

  desempenho = desempenho_modelo(nome_base_de_dados, best_model, nome_modelo, X_train, X_test, X_test_justica, y_pred, y_test, dados_sensiveis, printar=printar, matriz_de_confusao=matriz_de_confusao)
  desempenho['melhores_parametros'] = grid_search.best_params_

  # Importância das features

  best_params = {key.replace('classifier__estimator__', ''): value
                 for key, value in grid_search.best_params_.items()}

  # Instanciar um novo Perceptron com os MELHORES parâmetros
  final_inspection_model = Perceptron(**best_params)

  # Criar um pipeline de inspeção simples
  inspection_pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', final_inspection_model)
  ])

  # Treinar este pipeline nos dados de treino completos para obter os coeficientes finais
  inspection_pipeline.fit(X_train, y_train)

  # Agora, extrair os componentes deste pipeline treinado
  trained_linear_model = inspection_pipeline.named_steps['classifier']
  preprocessor_fitted = inspection_pipeline.named_steps['preprocessor']

  # Extrair coeficientes e nomes das features
  coefficients = trained_linear_model.coef_[0]
  feature_names = preprocessor_fitted.get_feature_names_out()

  # Criar e formatar o DataFrame de importância
  feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
  feature_importance['Magnitude'] = feature_importance['Importance'].abs()
  sorted_importance = feature_importance.sort_values(by='Magnitude', ascending=False)
  sorted_importance['Clean_Feature'] = sorted_importance['Feature'].str.replace(r'^(num__|remainder__)', '', regex=True)

  # Formatar a coluna Magnitude como string
  magnitude_str = sorted_importance['Magnitude'].map('{:.6f}'.format)

  # Aplicar a condição para indicar se o coeficiente é negativo
  condition = sorted_importance['Importance'] < 0
  value_if_true = magnitude_str + " (Negativo)"
  value_if_false = magnitude_str
  sorted_importance['Formatted_Importance'] = np.where(condition, value_if_true, value_if_false)

  # Imprimir a saída formatada
  output_string = sorted_importance[['Clean_Feature', 'Formatted_Importance']].to_string(
      index=False, header=False,
      formatters={'Clean_Feature': '{:<20}'.format, 'Formatted_Importance': '{:<25}'.format}
  )
  if(printar):
    print(f"\n----- {nome_base_de_dados} || IMPORTÂNCIA DAS FEATURES -----\n")
    print(output_string)

  desempenho["importancia_das_features"] = output_string

  return best_model, desempenho