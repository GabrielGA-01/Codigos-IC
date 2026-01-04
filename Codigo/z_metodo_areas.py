import pandas as pd
import numpy as np
import os
import glob

# Configuração para encontrar o arquivo mais recente automaticamente se o especificado não existir
caminho_resultados = "./Resultados"
nome_arquivo_excel = f"{caminho_resultados}/resultado_global_03_01_2026_19_50.xlsx"

if not os.path.exists(nome_arquivo_excel):
    # Tenta encontrar o arquivo mais recente
    list_of_files = glob.glob(f'{caminho_resultados}/resultado_global_*.xlsx')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Arquivo especificado não encontrado. Usando o mais recente: {latest_file}")
        nome_arquivo_excel = latest_file
    else:
        print(f"Erro: O arquivo '{nome_arquivo_excel}' não foi encontrado e nenhum outro arquivo compatível foi localizado.")
        exit()

# Nomes das planilhas (sheets) a serem lidas
sheet_resultados_completos = "resultados_completos"
sheet_fairness_results = "fairness_results"
sheet_extended_fairness_results = "extended_fairness_results"

SIN_120 = np.sqrt(3) / 2

# --- 1. CARREGAMENTO DOS DADOS DIRETAMENTE DO EXCEL ---
def clean_and_convert(df, cols):
    """Converte colunas para numérico, tratando possíveis erros de formatação."""
    for col in cols:
        # Usa errors='coerce' para transformar valores não numéricos (como strings ou NaN) em NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

try:
    print(f"Lendo o arquivo Excel: {nome_arquivo_excel}")
    
    # Leitura da planilha de resultados de desempenho (F1-Score)
    # Precisamos da coluna 'dataset' também agora
    df_comp = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_resultados_completos)
    df_comp = clean_and_convert(df_comp, ['f1_teste'])
    
    # Leitura da planilha de resultados de fairness estendida (EOR e DPR)
    # Precisamos da coluna 'dataset' também
    df_ext_fair = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_extended_fairness_results)
    df_ext_fair = clean_and_convert(df_ext_fair, ['equalized_odds_difference', 'demographic_parity_difference'])
    
    print("Arquivos carregados com sucesso.")

except Exception as e:
    print(f"Erro ao ler ou processar o arquivo Excel: {e}")
    exit()

# --- 2. SELEÇÃO E FILTRAGEM ---

# Seleciona apenas as colunas necessárias
# Assumindo que 'dataset' e 'tecnica' são as chaves de junção
cols_comp = ['dataset', 'tecnica', 'f1_teste']
cols_fair = ['dataset', 'tecnica', 'equalized_odds_difference', 'demographic_parity_difference']

df_comp = df_comp[cols_comp]
df_ext_fair = df_ext_fair[cols_fair]

# Join dos dataframes
# Agora mergeamos por 'dataset' E 'tecnica' para ter o resultado por dataset
df_analise = df_comp.merge(df_ext_fair, on=['dataset', 'tecnica'], how='inner').dropna()

# Renomear colunas para facilitar manipulação
df_analise = df_analise.rename(columns={
    'f1_teste': 'F1',
    'equalized_odds_difference': 'EOD',
    'demographic_parity_difference': 'DPD'
})

# --- 3. CÁLCULO DA MÉDIA POR TÉCNICA DENTRO DE CADA DATASET ---
# Agrupa por dataset e técnica e calcula a média das métricas (agregando todos os modelos)
df_analise = df_analise.groupby(['dataset', 'tecnica'])[['F1', 'EOD', 'DPD']].mean().reset_index()

# --- 4. CÁLCULO DA TÉCNICA (Métrica Composta) PARA CADA DATASET ---

# Passo 1: Transformar EOD e DPD para Proximidade ao Ideal (0.0)
# A diferença ideal é 0. Quanto mais longe de 0 (positivo ou negativo), pior.
# Proximidade = 1 - abs(Diferença)
df_analise['EOD_Prox'] = 1 - np.abs(df_analise['EOD'])
df_analise['DPD_Prox'] = 1 - np.abs(df_analise['DPD'])

# Tratar casos onde a diferença é muito grande (>1 ou <-1) e a proximidade fica negativa
# Podemos clipar em 0 (pessimo) e 1 (otimo)
df_analise['EOD_Prox'] = df_analise['EOD_Prox'].clip(lower=0)
df_analise['DPD_Prox'] = df_analise['DPD_Prox'].clip(lower=0)

# Simplifica variáveis
F1 = df_analise['F1']
EOR_P = df_analise['EOD_Prox'] # Mantendo nome de variavel similar para calculo da area
DPR_P = df_analise['DPD_Prox']

# Passo 2: Calcular a Soma das Áreas Par a Par * sin(120º)
# Métricas: F1, EOD_Imp, DPD_Imp
# Área Total = sin(120º) * [ (F1 * EOD_Imp) + (EOD_Imp * DPD_Imp) + (DPD_Imp * F1) ]
df_analise['MCPM'] = (F1 * EOR_P * SIN_120) + (EOR_P * DPR_P * SIN_120) + (DPR_P * F1 * SIN_120)

# --- 5. EXIBIÇÃO DOS RESULTADOS ---

# Identificar datasets "Não Enviesados" (os que contém "ORIGINAL" no nome) vs "Enviesados"
# Assumimos que o nome do dataset está na coluna 'dataset'
datasets_unicos = df_analise['dataset'].unique()

# Função de ordenação: Primeiro os que tem "ORIGINAL", depois alfabético
def criterio_ordenacao(nome):
    e_original = "ORIGINAL" in nome.upper()
    return (not e_original, nome) # False (0) vem antes de True (1), então 'not' faz Original vir antes

datasets_ordenados = sorted(datasets_unicos, key=criterio_ordenacao)

print("\nMétrica de Performance Multi-Critério (MCPM) - Por Dataset (Média das Técnicas)\n")
print(f"Métricas Utilizadas: F1-Score, Equalized Odds Diff (Prox. 0), Demographic Parity Diff (Prox. 0)")
print("=" * 100)

for dataset in datasets_ordenados:
    print(f"\nDATASET: {dataset}")
    print("-" * 100)
    
    # Filtrar e ordenar por MCPM
    df_subset = df_analise[df_analise['dataset'] == dataset].copy()
    df_subset = df_subset.sort_values(by='MCPM', ascending=False)
    
    # Seleção e formatação
    # O usuário pediu para apresentar como "1 - valor", que corresponde ao nosso cálculo de Proximidade (Prox)
    # Então substituímos EOD -> EOD_Prox e DPD -> DPD_Prox na saída
    cols_output = ['tecnica', 'F1', 'EOD_Prox', 'DPD_Prox', 'MCPM']
    df_subset = df_subset[cols_output]
    
    # Renomear para ficar claro na tabela
    df_subset = df_subset.rename(columns={
        'EOD_Prox': 'EOD (1-|Diff|)',
        'DPD_Prox': 'DPD (1-|Diff|)'
    })
    
    df_subset = df_subset.round(4)
    
    print(df_subset.to_markdown(index=False, numalign="center", stralign="left"))
    print("-" * 100)
