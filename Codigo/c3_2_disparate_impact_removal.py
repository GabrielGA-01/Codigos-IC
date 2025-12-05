import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import Transformer

# Adaptação para a técnica contendo o método fit() e transform()
class DisparateImpactRemover_Mod(Transformer):
    
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

    # Criação do metodo fit
    def fit(self, dataset):
        """Aprende o reparador a partir do dataset de treino."""
        if not self.sensitive_attribute:
            self.sensitive_attribute = dataset.protected_attribute_names[0]

        features = dataset.features.tolist()
        self.index_ = dataset.feature_names.index(self.sensitive_attribute)

        # cria e armazena o reparador (não aplica ainda)
        self.repairer_ = self.Repairer(features, self.index_, self.repair_level, False)
        self.fitted_ = True
        return self

    # Criação do método transform
    def transform(self, dataset):
        """Aplica o reparo usando o reparador já aprendido no fit()."""
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

    # --- Mantém compatibilidade com o padrão original ---
    def fit_transform(self, dataset):
        """Treina o reparador e aplica o reparo no mesmo dataset."""
        return self.fit(dataset).transform(dataset)
    
def disparate_impact_removal(funcao_modelo, parametros_in, printar=False):
    parametros = parametros_in.copy()
    dados_sensiveis = parametros['dados_sensiveis']

    # Ajustando o nome
    parametros['nome_base_de_dados'] = parametros['nome_base_de_dados'] + " || DISPARATE IMPACT REMOVAL"

    # Formando o dataset de treino
    dados_treino = pd.concat([parametros['X_train'], parametros['y_train']], axis=1)

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

    # Voltando para o dataframe
    df_repaired_train, _ = dataset_repaired_train.convert_to_dataframe()

    # Formando os dados de teste e aplicando o transform neles
    dados_teste = pd.concat([parametros['X_test'], parametros['y_test']], axis=1)

    dataset_test = BinaryLabelDataset(
        df=dados_teste,
        favorable_label=dados_sensiveis['rotulo_favoravel'],
        unfavorable_label=dados_sensiveis['rotulo_desfavoravel'],
        label_names=['target'],
        protected_attribute_names=[dados_sensiveis['coluna_sensivel']]
    )

    # Aplicando a transformação sem dar fit de novo
    dataset_repaired_test = DIR.transform(dataset_test)

    # Voltando para o dataframe
    df_repaired_test, _ = dataset_repaired_test.convert_to_dataframe()

    # Separando X_test e y_test do DataFrame reparado
    parametros['X_train'] = df_repaired_train.drop('target', axis=1)
    parametros['y_train'] = df_repaired_train['target']
    parametros['X_test'] = df_repaired_test.drop('target', axis=1)
    parametros['y_test'] = df_repaired_test['target']

    if(printar):
        print(f"\n----- {parametros['nome_base_de_dados']} || VISUALIZANDO AS MUDANÇAS -----\n")
        print("Visualização do DataFrame Original de Treino:")
        print(dados_treino.head())
        print("\nVisualização do DataFrame Reparado de Treino:")
        print(df_repaired_train.head())
        print("\nVisualização do DataFrame Original de Teste:")
        print(dados_teste.head())
        print("\nVisualização do DataFrame Reparado de Teste:")
        print(df_repaired_test.head())

    # TREINANDO UM MODELO

    X_test_justica = parametros['X_test'].copy()

    # Excluir a coluna sensível - Próprio da técnica
    parametros['X_train'] = parametros['X_train'].drop(dados_sensiveis['coluna_sensivel'], axis=1)
    parametros['X_test'] = parametros['X_test'].drop(dados_sensiveis['coluna_sensivel'], axis=1)
    if(printar):
        print(f"A coluna {dados_sensiveis['coluna_sensivel']} não foi usada no treinamento do modelo")

    # Parâmetros já contém os dados de treino e teste e o nome dos dados
    # Parâmetros específicos dessa técnica
    parametros['X_justica'] = True
    parametros['X_test_justica'] = X_test_justica

    _, desempenho = funcao_modelo(**parametros)

    return(desempenho)

if __name__ == "__main__":
    from c0_1_configuracoes import (
        param_grid_random_forest_basico,
        preprocessor_passthrough,
        dados_sensiveis_sexo
        )
    from c2_1_random_forest import random_forest_GSCV
    from c1_3_preprocessor_automatizado import aplicar_pre_processador

    df = pd.read_csv(f"Datasets/Processados/Dataset3.csv", sep=',')

    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    parametros= {}
    parametros['printar'] = True
    parametros['matriz_de_confusao'] = False
    parametros['param_grid'] = param_grid_random_forest_basico
    parametros['preprocessor'] = preprocessor_passthrough
    parametros['cv_n_splits'] = 2
    parametros['X_train'] = X_train
    parametros['X_test'] = X_test
    parametros['y_train'] = y_train
    parametros['y_test'] = y_test
    parametros['nome_base_de_dados'] = "DATASET"
    parametros['dados_sensiveis'] = dados_sensiveis_sexo

    desempenho = disparate_impact_removal(random_forest_GSCV, parametros, printar=True)
   