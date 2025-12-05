# Treina um modelo XGboost usando pipeline, grid search e cross validation

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

def xgboost_GSCV(param_grid, preprocessor, cv_n_splits, nome_base_de_dados, X_train, X_test, y_train, y_test, dados_sensiveis, printar=False, matriz_de_confusao=False, grafico_shap=False, pesos=False, pesos_modelo=None, X_justica=False, X_test_justica=None):
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

  desempenho = desempenho_modelo(nome_base_de_dados, best_model, X_test, X_test_justica, y_pred, y_test, dados_sensiveis, printar=printar, matriz_de_confusao=matriz_de_confusao)
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