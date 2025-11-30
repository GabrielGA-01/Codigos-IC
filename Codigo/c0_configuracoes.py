# Esse aquivo tem o propósito de conter variáveis globais
# para serem utilizadas por todos os notebooks e facilitar
# assim, mudanças futuras

seed = 19
caminho_original = 'Datasets/Originais'
caminho_processado = 'Datasets/Processados'

# PARÂMETROS GRID SEARCH

param_grid_regressao_logistica_completo = {
  'classifier__random_state': [seed],
  'classifier__penalty': ['l2'],
  'classifier__solver': ['lbfgs'],
  'classifier__max_iter': [1000],
  'classifier__C': [0.1, 1, 10]
}
param_grid_perceptron_completo = {
    'classifier__estimator__random_state': [seed],
    'classifier__estimator__penalty': ['l2', 'l1', 'elasticnet', None],
    'classifier__estimator__alpha': [0.0001, 0.001, 0.01, 0.1],
    'classifier__estimator__max_iter': [1000, 2000, 3000],
    'classifier__estimator__eta0': [0.1, 0.01, 0.001],
    'classifier__estimator__tol': [1e-3, 1e-4]
}
param_grid_random_forest_completo = {
    'classifier__random_state': [seed],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5], 
    'classifier__min_samples_leaf': [1, 2, 3], 
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features': ['sqrt', 'log2'] 
}
param_grid_xgboost_completo = {
  'classifier__random_state': [seed],
  'classifier__eval_metric': ['logloss'],
  'classifier__learning_rate': [0.01, 0.05, 0.1],
  'classifier__max_depth': [3, 5, 7],
  'classifier__n_estimators': [50, 100, 200],
  'classifier__subsample': [0.6, 0.8, 1.0],
  'classifier__colsample_bytree': [0.6, 0.8, 1.0]
}

param_grid_regressao_logistica_basico = {
    'classifier__random_state': [seed],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs'],
    'classifier__max_iter': [1000],
    'classifier__C': [1]
}
param_grid_perceptron_basico = {
    'classifier__estimator__random_state': [seed],
    'classifier__estimator__eta0': [0.1],
    'classifier__estimator__max_iter': [200]
}
param_grid_random_forest_basico = {
    'classifier__random_state': [seed],
    'classifier__n_estimators': [100],
    'classifier__max_depth': [10]
}
param_grid_xgboost_basico = {
    'classifier__random_state': [seed],
    'classifier__eval_metric': ['logloss'],
    'classifier__learning_rate': [0.1],
    'classifier__max_depth': [5],
    'classifier__n_estimators': [100], 
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [1] 
}