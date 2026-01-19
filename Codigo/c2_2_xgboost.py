"""
Este arquivo contém a implementação do treinamento do modelo XGBoost utilizando 
Pipeline, GridSearchCV e StratifiedKFold para otimização de hiperparâmetros.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)
import xgboost as xgb

from c0_1_configuracoes import seed
from c1_4_desempenho_modelos import desempenho_modelo
from c1_5_grafico_shap_violino import fazer_grafico_shap

def xgboost_GSCV(param_grid, preprocessor, cv_n_splits, nome_base_de_dados, X_train, X_test, y_train, y_test, dados_sensiveis, printar=False, matriz_de_confusao=False, grafico_shap=False, pesos=False, pesos_modelo=None, X_justica=False, X_test_justica=None, nome_modelo='xgboost'):
  """
  Executa a busca de hiperparâmetros e treina o XGBoost, avaliando seu desempenho e importância das features.
  
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
  - grafico_shap: Booleano para gerar gráfico de violino SHAP.
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

  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', xgb.XGBClassifier())
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

  if(grafico_shap):
    fazer_grafico_shap(nome_base_de_dados, best_model, X_test)

  desempenho = desempenho_modelo(nome_base_de_dados, best_model, nome_modelo, X_train, X_test, X_test_justica, y_pred, y_test, dados_sensiveis, printar=printar, matriz_de_confusao=matriz_de_confusao)
  desempenho['melhores_parametros'] = grid_search.best_params_

  # Importância das features
  feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

  feature_importances = pd.Series(
      best_model.named_steps['classifier'].feature_importances_,
      index=feature_names
  ).sort_values(ascending=False)

  desempenho["importancia_das_features"] = feature_importances

  # Importância das features
  if(printar):
    print(f"\n----- {nome_base_de_dados} || IMPORTÂNCIA DAS FEATURES -----\n")
    print(feature_importances)

  return best_model, desempenho