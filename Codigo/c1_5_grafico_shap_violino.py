"""
Este arquivo contém a função para gerar um gráfico de violino baseado nos valores SHAP,
permitindo visualizar o impacto das features para diferentes grupos sensíveis (ex: Masculino vs Feminino).
Atualmente está em desuso e precisa ser atualizada.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

def fazer_grafico_shap(nome_base_de_dados, trained_model_pipeline, X_test):
    """
    Calcula valores SHAP e plota um gráfico de violino comparativo por sexo.
    
    Parâmetros:
    - nome_base_de_dados: String com o nome do dataset.
    - trained_model_pipeline: Objeto Pipeline treinado contendo o classificador.
    - X_test: DataFrame contendo os dados de teste para os quais os valores SHAP serão calculados.
    """
    print(f"\n----- {nome_base_de_dados} || GRÁFICO SHAP -----\n")
    model = trained_model_pipeline.named_steps['classifier']

    explainer = shap.TreeExplainer(model)

    # Calculo dos valores SHAP
    explanation = explainer(X_test)

    # Selecionando os valores para a classe positiva (1)
    shap_values_class1 = explanation.values

    df_shap = pd.DataFrame(shap_values_class1, columns=X_test.columns)

    df_features = X_test.reset_index(drop=True)

    df_long = pd.melt(df_shap, var_name='Feature', value_name='SHAP Value')

    sensitive_col = np.tile(df_features['SEX_MALE'].values, len(X_test.columns))
    df_long['SEX_MALE'] = sensitive_col

    # Criação do Gráfico de Violino Dividido
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(24, 6))

    sns.violinplot(
        data=df_long,
        x='Feature',
        y='SHAP Value',
        hue='SEX_MALE',    # A coluna que vai dividir
        split=True,        # Divide no meio
        inner='quart',     # Mostra os quartis
        palette={0: '#4c72b0', 1: '#dd8452'}
    )

    # Configurações do gráfico
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribuição dos Valores SHAP por Feature e Gênero', fontsize=16)
    plt.xlabel('')

    # Configuração da legenda
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Feminino', 'Masculino'], title='Demographic Group')

    plt.tight_layout()
    plt.show()