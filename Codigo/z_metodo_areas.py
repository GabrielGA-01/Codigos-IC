import pandas as pd
import numpy as np

# Nome do arquivo Excel original
nome_arquivo_excel = "resultado_global_08_12_2025_18.xlsx"

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
    
    # Leitura da planilha de resultados de desempenho
    df_comp = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_resultados_completos)
    df_comp = clean_and_convert(df_comp, ['acuracia_teste', 'auc_teste', 'f1_teste'])

    # Leitura da planilha de resultados de fairness
    df_fair = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_fairness_results)
    df_fair = clean_and_convert(df_fair, ['true_positive_rate', 'true_negative_rate', 'false_positive_rate', 'false_negative_rate'])

    # Leitura da planilha de resultados de fairness estendida (para Predictive Parity)
    df_ext_fair = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_extended_fairness_results)
    df_ext_fair = clean_and_convert(df_ext_fair, ['predictive_parity_ratio'])
    
    print("Arquivos carregados com sucesso.")

except FileNotFoundError:
    print(f"❌ Erro: O arquivo '{nome_arquivo_excel}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"❌ Erro ao ler ou processar o arquivo Excel: {e}")
    exit()

# --- 2. CÁLCULO DAS MÉDIAS POR TÉCNICA ---

# Médias de Desempenho
avg_comp = df_comp.groupby('tecnica')[['acuracia_teste', 'f1_teste']].mean().rename(columns={
    'acuracia_teste': 'Acurácia', 'f1_teste': 'F1-Score'
})

# Média do Equalized Odds Ratio (EOR)
avg_eor = df_ext_fair.groupby('tecnica')['equalized_odds_ratio'].mean().to_frame().rename(columns={
    'equalized_odds_ratio': 'EOR Média'
})

# Combinação dos resultados médios
df_analise = avg_comp.merge(avg_eor, on='tecnica', how='inner').dropna()

# --- 3. TRANSFORMAÇÃO E CÁLCULO DA ÁREA TOTAL (Métrica Composta) ---

# Passo 1: Transformar EOR Média para Proximidade ao Ideal (1.0)
df_analise['EOR Proximidade'] = 1 - np.abs(df_analise['EOR Média'] - 1.0)

# Simplifica os nomes das colunas para o cálculo
A = df_analise['Acurácia']
F = df_analise['F1-Score']
E = df_analise['EOR Proximidade']

# Passo 2: Calcular a Soma das Áreas Par a Par * sin(120º)
# Fórmula: Area Total = sin(120º) * [ (A*F) + (A*E) + (F*E) ]
df_analise['Área Total (120º)'] = (A * F * SIN_120) + (A * E * SIN_120) + (F * E * SIN_120)

# Passo 3: Calcular a Área Média
df_analise['Área Média (120º - Par a Par)'] = df_analise['Área Total (120º)'] / 3.0


# --- 4. EXIBIÇÃO DOS RESULTADOS ---
df_final = df_analise.reset_index()

# Seleção e formatação para a saída
df_final = df_final[['tecnica', 'Acurácia', 'F1-Score', 'EOR Média', 'Área Média (120º - Par a Par)']]
df_final.columns = ['Técnica', 'Acurácia', 'F1-Score', 'EOR (Média)', 'MCPM (Área Média 120º)']
df_final[['Acurácia', 'F1-Score', 'EOR (Média)', 'MCPM (Área Média 120º)']] = df_final[['Acurácia', 'F1-Score', 'EOR (Média)', 'MCPM (Área Média 120º)']].round(4)
df_final = df_final.sort_values(by='MCPM (Área Média 120º)', ascending=False)

print("\n## 🌐 Métrica de Performance Multi-Critério (MCPM)\n")
print(df_final.to_markdown(index=False, numalign="center", stralign="left"))