# Esse aquivo tem o propósito de conter variáveis globais
# para serem utilizadas por todos os notebooks e facilitar
# assim, mudanças futuras

from sklearn.compose import ColumnTransformer



seed = 19
caminho_original = 'Datasets/Originais'
caminho_processado = 'Datasets/Processados'

# Pré-processador que não faz nada
preprocessor_passthrough = ColumnTransformer(
    transformers=[],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# DADOS SENSÍVEIS

# Atualmente todos os quatro datasets têm rótulo favorável como 0 e desfavorável como 1.
# Futuramente isso pode mudar e pode ser necessário criar variações dos dicionários abaixo.

dados_sensiveis_sexo = {}
dados_sensiveis_sexo['coluna_sensivel'] = 'sensitive_sexo'
dados_sensiveis_sexo['grupo_privilegiado'] =  1     # Homens
dados_sensiveis_sexo['grupo_desprivilegiado'] = 0   # Mulheres
dados_sensiveis_sexo['rotulo_favoravel'] = 0 
dados_sensiveis_sexo['rotulo_desfavoravel'] = 1

dados_sensiveis_age = {}
dados_sensiveis_age['coluna_sensivel'] = 'sensitive_age'
dados_sensiveis_age['grupo_privilegiado'] =  1      # Adultos
dados_sensiveis_age['grupo_desprivilegiado'] = 0    # Jovens
dados_sensiveis_age['rotulo_favoravel'] = 0
dados_sensiveis_age['rotulo_desfavoravel'] = 1

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

# COLUNAS DISCRETAS PARA O SMOTENC
# Observação: para utilizar é preciso adicionar apenas um dos dois:
# - 'sensitive_sexo'
# - 'sensitive_age'

colunas_discretas1 = [
    'IS_MARRIAGED', 'EDUCATION', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 
    'PAY_5', 'PAY_6'
]

colunas_discretas2 = [
    'children_cat_one_child', 'children_cat_two_or_more',
    'fam_size_cat_three_or_more', 'fam_size_cat_two_members',
    'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant',
    'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Working',
    'NAME_HOUSING_TYPE_House / apartment', 'NAME_HOUSING_TYPE_Municipal apartment',
    'NAME_HOUSING_TYPE_Office apartment', 'NAME_HOUSING_TYPE_Rented apartment',
    'NAME_HOUSING_TYPE_With parents', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
    'ordinal_EDUCATION', 'IS_MARRIAGE', 'IS_UNEMPLOYED'
]

colunas_discretas3 = [
    'Marital_Status_Married', 'Marital_Status_Single',
    'Marital_Status_Unknown', 'Education_Level', 'Income_Category', 
    'Card_Category'
]

colunas_discretas4 = [
    'Attribute1_A12', 'Attribute1_A13', 'Attribute1_A14', 'Attribute3_A31',
    'Attribute3_A32', 'Attribute3_A33', 'Attribute3_A34', 'Attribute4_A41',
    'Attribute4_A410', 'Attribute4_A42', 'Attribute4_A43', 'Attribute4_A44',
    'Attribute4_A45', 'Attribute4_A46', 'Attribute4_A48', 'Attribute4_A49',
    'Attribute6_A62', 'Attribute6_A63', 'Attribute6_A64', 'Attribute6_A65',
    'Attribute7_A72', 'Attribute7_A73', 'Attribute7_A74', 'Attribute7_A75',
    'Attribute10_A102', 'Attribute10_A103', 'Attribute12_A122', 'Attribute12_A123',
    'Attribute12_A124', 'Attribute14_A142', 'Attribute14_A143', 'Attribute15_A152',
    'Attribute15_A153', 'Attribute17_A172', 'Attribute17_A173', 'Attribute17_A174',
    'Attribute19_A192', 'Attribute20_A202', 'ordinal_Attribute8', 'ordinal_Attribute11',
    'ordinal_Attribute16', 'ordinal_Attribute18'
]