import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Tipo de variáveis
# =====================

def listar_variaveis_por_tipo(df):
    # Colunas por tipo (considera também dtypes "nullable" do pandas)
    obj_cols   = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols   = df.select_dtypes(include=['category']).columns.tolist()
    int_cols   = df.select_dtypes(include=['int64', 'int32', 'Int64', 'Int32']).columns.tolist()
    float_cols = df.select_dtypes(include=['float64', 'float32', 'Float64', 'Float32']).columns.tolist()

    # Impressão no formato solicitado
    print("variaveis 'object','category' =", obj_cols + cat_cols)
    print("variaveis 'int64','float64'  =", int_cols + float_cols)

    # (Opcional) retornar um dicionário para uso programático
    return {
        "object_category": obj_cols + cat_cols,
        "int": int_cols,
        "float": float_cols
    }

# =====================
# Analisar variáveis uma a uma
# =====================
def analisar_variavel(df, col):
    print("="*60)
    print(f"📊 Variável: {col}")
    print("Tipo:", df[col].dtype)
    print("Nulos:", df[col].isna().sum())
    print("Valores únicos:", df[col].nunique())
    print("\nTop valores:")
    print(df[col].value_counts(normalize=True).head(10))
    
    if pd.api.types.is_numeric_dtype(df[col]):
        print("\nEstatísticas descritivas:")
        print(df[col].describe())


# =====================
# Analisar distribuição de target e sexo
# =====================

def plot_sex_distribution(df, title):
    # Substituir valores numéricos por rótulos textuais
    df['sensitive_sexo'] = df['sensitive_sexo'].map({1: 'Male', 0: 'Female'})

    # Criar o gráfico de contagem
    ax = sns.countplot(data=df, x='sensitive_sexo', hue='target')

    # Adicionar rótulos de contagem nas barras
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fontsize=8)

    # Títulos e rótulos
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('sensitive_sexo')
    ax.set_ylabel('Count')

    plt.show()

def plot_age_distribution(df, title):
    # Garantir que a coluna é categórica na ordem desejada
    categorias = ["jovem", "adulto", "idoso"]
    df["sensitive_age"] = df["sensitive_age"].astype("category")
    df["sensitive_age"] = df["sensitive_age"].cat.set_categories(categorias, ordered=True)

    # Gráfico
    ax = sns.countplot(data=df, x='sensitive_age', hue='target')

    # Labels nas barras
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fontsize=8)

    # Títulos e rótulos
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('sensitive_age')
    ax.set_ylabel('Count')

    plt.show()