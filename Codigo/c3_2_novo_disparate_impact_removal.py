"""
Este arquivo implementa a técnica de pré-processamento Disparate Impact Removal (AIF360).
A técnica modifica os valores das features para remover a correlação com o atributo sensível.
Inclui uma classe personalizada para permitir o uso separado de fit() e transform().
"""

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import Transformer

class DisparateImpactRemover_Mod(Transformer):
    """
    Versão modificada do DisparateImpactRemover para garantir que o reparador treinado no 
    conjunto de treino possa ser aplicado corretamente ao conjunto de teste.
    """
    
    def __init__(self, repair_level=1.0, sensitive_attribute=''):
        super(DisparateImpactRemover_Mod, self).__init__(repair_level=repair_level)
        from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
        self.Repairer = Repairer

        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")

        self.repair_level = repair_level
        self.sensitive_attribute = sensitive_attribute
        self.repairer_ = None      # armazenará o reparador treinado
        self.index_ = None         # índice do atributo sensível
        self.fitted_ = False       # flag para saber se o fit foi feito

    def fit(self, dataset):
        """Aprende o reparador (Repairer) a partir do dataset fornecido."""
        if not self.sensitive_attribute:
            self.sensitive_attribute = dataset.protected_attribute_names[0]

        features = dataset.features.tolist()
        self.index_ = dataset.feature_names.index(self.sensitive_attribute)

        # cria e armazena o reparador (não aplica ainda)
        self.repairer_ = self.Repairer(features, self.index_, self.repair_level, False)
        self.fitted_ = True
        return self

    def transform(self, dataset):
        """Aplica a transformação de reparo utilizando os coeficientes aprendidos no fit()."""
        if not self.fitted_:
            raise RuntimeError("Você precisa chamar fit() antes de transform().")

        features = dataset.features.tolist()
        repaired = dataset.copy()

        # usa o mesmo reparador treinado no fit()
        repaired_features = self.repairer_.repair(features)
        repaired.features = np.array(repaired_features, dtype=np.float64)

        # garante que o atributo sensível original não seja alterado
        idx_sa = dataset.protected_attribute_names.index(self.sensitive_attribute)
        repaired.features[:, self.index_] = repaired.protected_attributes[:, idx_sa]

        return repaired

    def fit_transform(self, dataset):
        """Treina e transforma o dataset simultaneamente."""
        return self.fit(dataset).transform(dataset)
    




def disparate_impact_removal(parametros_dataset, dados_dataset_in, parametros_algoritmos, printar=False):
    # Para não alterar os dados originais
    dados_dataset = dados_dataset_in.copy()
    nome_do_teste = dados_dataset['nome_base_de_dados'] + " || DISPARATE IMPACT REMOVAL"

    # Criando uma variável para os dados sensíveis
    dados_sensiveis = parametros_dataset['dados_sensiveis']

    # Formando o dataset de treino
    dados_treino = pd.concat([dados_dataset['X_train'], dados_dataset['y_train']], axis=1)

    # Convertendo o DataFrame do Pandas para o formato do AIF360
    dataset_train = BinaryLabelDataset(
        df=dados_treino,
        favorable_label=dados_sensiveis['rotulo_favoravel'],
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'],
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )

    # Aplicando a técnica
    DIR = DisparateImpactRemover_Mod(repair_level=1.0)
    dataset_repaired_train = DIR.fit_transform(dataset_train)

    # Voltando para o formato dataframe
    df_repaired_train, _ = dataset_repaired_train.convert_to_dataframe()

    # Formando os dados de teste e aplicando o transform neles
    dados_teste = pd.concat([dados_dataset['X_test'], dados_dataset['y_test']], axis=1)

    dataset_test = BinaryLabelDataset(
        df=dados_teste,
        favorable_label=dados_sensiveis['rotulo_favoravel'],
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'],
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )

    # Aplicando a transformação sem dar fit de novo
    dataset_repaired_test = DIR.transform(dataset_test)

    # Voltando para o formato dataframe
    df_repaired_test, _ = dataset_repaired_test.convert_to_dataframe()

    # Separando X_test e y_test do DataFrame reparado
    dados_dataset['X_train'] = df_repaired_train.drop('target', axis=1)
    dados_dataset['y_train'] = df_repaired_train['target']
    dados_dataset['X_test'] = df_repaired_test.drop('target', axis=1)
    dados_dataset['y_test'] = df_repaired_test['target']

    if(printar):
        print(f"\n----- {nome_do_teste} || VISUALIZANDO AS MUDANÇAS -----\n")
        print("Visualização do DataFrame Original de Treino:")
        print(dados_treino.head())
        print("\nVisualização do DataFrame Reparado de Treino:")
        print(df_repaired_train.head())
        print("\nVisualização do DataFrame Original de Teste:")
        print(dados_teste.head())
        print("\nVisualização do DataFrame Reparado de Teste:")
        print(df_repaired_test.head())

    # Excluir a coluna sensível - Próprio da técnica
    X_test_justica = dados_dataset['X_test'].copy()
    
    dados_dataset['X_train'] = dados_dataset['X_train'].drop(dados_sensiveis['coluna_sensivel'], axis=1)
    dados_dataset['X_test'] = dados_dataset['X_test'].drop(dados_sensiveis['coluna_sensivel'], axis=1)
    if(printar):
        print(f"A coluna {dados_sensiveis['coluna_sensivel']} não foi usada no treinamento do modelo")

    # Concatenando dicionários para usar no treinamento do modelo
    parametros = parametros_dataset | dados_dataset

    # Parâmtros específico da técnica
    parametros['X_justica'] = True
    parametros['X_test_justica'] = X_test_justica

    desempenho = {}
    for nome_algoritmo in parametros_algoritmos:
        algoritmo = parametros_algoritmos[nome_algoritmo]

        parametros['param_grid'] = algoritmo['parametros_grid']
        parametros['nome_base_de_dados'] = nome_do_teste + f" || {nome_algoritmo.upper().replace('_', ' ')}"
        _, desempenho[nome_algoritmo] = algoritmo['funcao'](**parametros)

    return(desempenho)





if __name__ == "__main__":
    from c0_1_configuracoes import (
        param_grid_random_forest_basico,
        param_grid_xgboost_basico,
        param_grid_regressao_logistica_basico,
        param_grid_perceptron_basico,
        preprocessor_passthrough,
        dados_sensiveis_sexo,
        dados_sensiveis_age,
        )
    from c1_3_preprocessor_automatizado import aplicar_pre_processador
    from c2_1_random_forest import random_forest_GSCV
    from c2_2_xgboost import xgboost_GSCV
    from c2_3_regressao_logistica import regressao_logistica_GSCV
    from c2_4_perceptron import perceptron_GSCV

    df = pd.read_csv(f"Datasets/Processados/Dataset3.csv", sep=',')

    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    parametros_dataset = {}
    parametros_dataset['printar'] = True
    parametros_dataset['preprocessor'] = preprocessor_passthrough
    parametros_dataset['cv_n_splits'] = 2
    parametros_dataset['dados_sensiveis'] = dados_sensiveis_sexo
    
    dados_dataset = {}
    dados_dataset['X_train'] = X_train
    dados_dataset['X_test'] = X_test
    dados_dataset['y_train'] = y_train
    dados_dataset['y_test'] = y_test
    dados_dataset['nome_base_de_dados'] = "DATASET"

    parametros_random_forest = {
        'funcao': random_forest_GSCV,
        'parametros_grid': param_grid_random_forest_basico
    }
    parametros_xgboost = {
        'funcao': xgboost_GSCV,
        'parametros_grid': param_grid_xgboost_basico
    }
    parametros_regressao_logistica = {
        'funcao': regressao_logistica_GSCV,
        'parametros_grid': param_grid_regressao_logistica_basico
    }
    parametros_perceptron = {
        'funcao': perceptron_GSCV,
        'parametros_grid': param_grid_perceptron_basico
    }
    parametros_algoritmos = {
        'xgboost': parametros_xgboost,
        'random_forest': parametros_random_forest,
        'regressao_logistica': parametros_regressao_logistica,
        'perceptron': parametros_perceptron
    }

    desempenho = disparate_impact_removal(parametros_dataset, dados_dataset, parametros_algoritmos, True)

    print(desempenho)