from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
import pandas as pd

def instance_reweighing(funcao_modelo, parametros_in, printar=False):
    parametros = parametros_in.copy()
    dados_sensiveis = parametros['dados_sensiveis']

    # Ajustando o nome
    parametros['nome_base_de_dados'] = parametros['nome_base_de_dados'] + " || INSTANCE REWEIGHING"

    # Definindo os grupos privilegiados e desprivilegiados (lista de dicionário)
    privileged_groups = [{dados_sensiveis['coluna_sensivel']: dados_sensiveis['grupo_privilegiado']}]
    unprivileged_groups = [{dados_sensiveis['coluna_sensivel']: dados_sensiveis['grupo_desprivilegiado']}]

    # Formando o dataset de treino
    dados_treino = pd.concat([parametros['X_train'], parametros['y_train']], axis=1)

    # Convertendo o DataFrame do Pandas para o formato do AIF360
    dataset_orig = BinaryLabelDataset(
        df=dados_treino,
        favorable_label=dados_sensiveis['rotulo_favoravel'],
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'],
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)

    dataset_transf = RW.fit_transform(dataset_orig)
    weights = pd.Series(dataset_transf.instance_weights, index=dados_treino.index)

    # Resultados
    temp_df = dados_treino.copy()
    temp_df['weight'] = weights
    pesos_por_classe = temp_df.groupby([dados_sensiveis['coluna_sensivel'], 'target'])['weight'].mean()
    if(printar):
        print(f"\n----- {parametros['nome_base_de_dados']} || PESOS CALCULADOS -----\n")
        print(pesos_por_classe)

    # TREINANDO UM MODELO

    # Parâmetros já contém os dados de treino e teste e o nome dos dados
    # Parâmetros específicos dessa técnica
    parametros['pesos'] = True
    parametros['pesos_modelo'] = weights

    _, desempenho = funcao_modelo(**parametros)
    desempenho["pesos_das_classes"] = pesos_por_classe

    return(desempenho)

if __name__ == "__main__":
    from c0_1_configuracoes import (
        param_grid_xgboost_basico,
        preprocessor_passthrough,
        dados_sensiveis_sexo
        )
    from c2_2_xgboost import xgboost_GSCV
    from c1_3_preprocessor_automatizado import aplicar_pre_processador

    df = pd.read_csv(f"Datasets/Processados/Dataset4.csv", sep=',')

    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    parametros= {}
    parametros['printar'] = True
    parametros['matriz_de_confusao'] = False
    parametros['param_grid'] = param_grid_xgboost_basico
    parametros['preprocessor'] = preprocessor_passthrough
    parametros['cv_n_splits'] = 2
    parametros['X_train'] = X_train
    parametros['X_test'] = X_test
    parametros['y_train'] = y_train
    parametros['y_test'] = y_test
    parametros['nome_base_de_dados'] = "DATASET"
    parametros['dados_sensiveis'] = dados_sensiveis_sexo

    desempenho = instance_reweighing(xgboost_GSCV, parametros, printar=True)