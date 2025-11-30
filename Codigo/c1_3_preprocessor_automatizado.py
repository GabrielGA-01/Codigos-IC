# Esse arquivo contém os pré-processadores necessários para cada um dos datasets.
# Além disso, contém a função responsável por separar os dados em treino e teste,
# aplicar automaticamente o pré-processador correto e retornar os dados tratados.
# Treina-se o pré-processador nos dados de treino e aplica nos dados de treino e 
# teste.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

from c0_configuracoes import seed

def aplicar_pre_processador(conjunto_dados):
    # PRÉ-PROCESSADORES 

    # Dataset 1
    numerical_cols1 = [
      'LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    nominal_cols1 = []
    preprocessor1 = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numerical_cols1),
          ('nom', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), nominal_cols1)
      ],
      remainder='passthrough'
    )
    colunas1 = numerical_cols1+nominal_cols1

    # Dataset 2
    numerical_cols2 = ['AMT_INCOME_TOTAL']
    nominal_cols2 = ['children_cat','fam_size_cat', 'NAME_INCOME_TYPE','NAME_HOUSING_TYPE']
    preprocessor2 = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numerical_cols2),
          ('nom', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), nominal_cols2)
      ],
      remainder='passthrough'
    )
    colunas2 = numerical_cols2+nominal_cols2

    # Dataset 3
    numerical_cols3 = [
        'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
        'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Total_Revolving_Bal'
    ]
    nominal_cols3 = ['Marital_Status']
    ordinal_cols3 = ['Education_Level', 'Income_Category', 'Card_Category']
    education_order = ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']
    income_order = ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
    card_order = ['Blue', 'Silver', 'Gold', 'Platinum']
    preprocessor3 = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numerical_cols3),
          ('nom', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), nominal_cols3),
          ('ord', OrdinalEncoder(categories=[education_order, income_order, card_order], handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols3)
      ],
      remainder='passthrough'
    )
    colunas3 = numerical_cols3+nominal_cols3+ordinal_cols3

    # Dataset 4
    numerical_cols4 = ['Attribute2', 'Attribute5']
    categorical_cols4 = ['Attribute1', 'Attribute3', 'Attribute4', 'Attribute6', 'Attribute7', 'Attribute10', 
                        'Attribute12', 'Attribute14', 'Attribute15', 'Attribute17', 'Attribute19', 'Attribute20']
    preprocessor4 = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numerical_cols4),
          ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols4)
      ],
      remainder='passthrough',
      verbose_feature_names_out=False
    )
    colunas4 = numerical_cols4+categorical_cols4

    # LÓGICA DA FUNÇÃO

    # Converter as colunas do dataframe atual para um set para comparação
    set_colunas_dados = set(conjunto_dados.columns)

    preprocessor = None
    num_dataset_identificado = 0

    # Verifica se as colunas dos pré-processadores estão presentes no conjunto de dados
    if set(colunas1).issubset(set_colunas_dados):
        preprocessor = preprocessor1
        num_dataset_identificado = 1
    
    elif set(colunas2).issubset(set_colunas_dados):
        preprocessor = preprocessor2
        num_dataset_identificado = 2
        
    elif set(colunas3).issubset(set_colunas_dados):
        preprocessor = preprocessor3
        num_dataset_identificado = 3
        
    elif set(colunas4).issubset(set_colunas_dados):
        preprocessor = preprocessor4
        num_dataset_identificado = 4
        
    else:
        # Levanta um erro listando as colunas que vieram para ajudar no debug
        raise ValueError(
            f"Não foi possível identificar o dataset com base nas colunas fornecidas.\n"
            f"Colunas presentes no DataFrame recebido: {list(conjunto_dados.columns)}"
        )
    
    print(f"Dataset {num_dataset_identificado} identificado. Aplicando preprocessor correspondente.")
    # Separar as features e o rótulo
    X = conjunto_dados.drop('target', axis=1)
    y = conjunto_dados['target']

    # Separar os dados no conjunto de treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    # Treinando o pré-processador nas features de treino
    preprocessor.fit(X_train)

    # Aplicando o pré-processador
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Criando DataFrames para os dados processados de treino. 
    X_train_processed_df = pd.DataFrame(X_train_processed, index=X_train.index, columns=preprocessor.get_feature_names_out())
    X_test_processed_df = pd.DataFrame(X_test_processed, index=X_test.index, columns=preprocessor.get_feature_names_out())
    # Os rótulos são binários, logo não é preciso usar o pré-processador

    return X_train_processed_df, X_test_processed_df, y_train, y_test

# Exemplo de uso
if __name__ == "__main__":
    from c0_configuracoes import caminho_processado
    from c1_0_funcoes_analise_Julia import analisar_variavel

    df = pd.read_csv(f"{caminho_processado}/Dataset4.csv", sep=',')
    X_train, X_test, y_train, y_test = aplicar_pre_processador(df)

    df_treino = pd.concat([X_train, y_train], axis=1)

    for x in df_treino.columns:
      analisar_variavel(df_treino, x)