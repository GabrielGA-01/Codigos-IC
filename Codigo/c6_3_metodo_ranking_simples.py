"""
Este arquivo implementa um método de ranking simples para comparar as técnicas.
Ele atribui pontuações para diversas métricas de desempenho e justiça (1 a N) para,
ao final, identificar os melhores métodos em média.
"""

import pandas as pd
import numpy as np

# Nome do arquivo Excel original
nome_arquivo_excel = "./Resultados/Mesclagem_Resultados_Final.xlsx"

# Nomes das planilhas (sheets) a serem lidas
sheet_resultados_completos = "resultados_completos"
sheet_fairness_results = "fairness_results"
sheet_extended_fairness_results = "extended_fairness_results"

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
    print(f"Erro: O arquivo '{nome_arquivo_excel}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao ler ou processar o arquivo Excel: {e}")
    exit()


# --- 2. CÁLCULO DAS MÉDIAS POR TÉCNICA ---

# Médias de Desempenho (Acurácia, AUC, F1)
avg_comp = df_comp.groupby('tecnica')[['acuracia_teste', 'auc_teste', 'f1_teste']].mean().rename(columns={
    'acuracia_teste': 'Acurácia', 'auc_teste': 'AUC', 'f1_teste': 'F1'
})

# Médias de Taxas de Classificação (TPR, TNR, FPR, FNR)
avg_fair = df_fair.groupby('tecnica')[['true_positive_rate', 'true_negative_rate', 'false_positive_rate', 'false_negative_rate']].mean().rename(columns={
    'true_positive_rate': 'TPR', 'true_negative_rate': 'TNR', 'false_positive_rate': 'FPR', 'false_negative_rate': 'FNR'
})

# Média da Métrica de Justiça (Predictive Parity Ratio)
avg_ext_fair = df_ext_fair.groupby('tecnica')['predictive_parity_ratio'].mean().to_frame().rename(columns={
    'predictive_parity_ratio': 'Predictive Parity Ratio'
})

# Combinação dos resultados médios
results_df = avg_comp.merge(avg_fair, on='tecnica', how='outer').merge(avg_ext_fair, on='tecnica', how='outer').dropna() 

# --- 3. APLICAÇÃO DA PONTUAÇÃO (1=Pior, 8=Melhor) ---
metrics_higher_better = ['Acurácia', 'AUC', 'F1', 'TPR', 'TNR']
metrics_lower_better = ['FPR', 'FNR']
metric_pp = 'Predictive Parity Ratio'
max_rank = results_df['Acurácia'].rank(method='dense').max() 

# Pontuação para métricas onde 'Maior é Melhor'
for metric in metrics_higher_better:
    pandas_rank = results_df[metric].rank(method='dense', ascending=False)
    results_df[f'Score {metric}'] = (max_rank + 1 - pandas_rank).astype(int)

# Pontuação para métricas onde 'Menor é Melhor'
for metric in metrics_lower_better:
    pandas_rank = results_df[metric].rank(method='dense', ascending=True)
    results_df[f'Score {metric}'] = (max_rank + 1 - pandas_rank).astype(int)

# Pontuação para Predictive Parity Ratio (O mais próximo de 1.0 é o melhor)
# Usa a Proximidade (1 - |Valor - 1.0|), onde Maior Proximidade é Melhor.
results_df['PP_Proximity'] = 1 - np.abs(results_df[metric_pp] - 1.0)
pandas_rank_pp = results_df['PP_Proximity'].rank(method='dense', ascending=False)
results_df[f'Score {metric_pp}'] = (max_rank + 1 - pandas_rank_pp).astype(int)
results_df = results_df.drop(columns=['PP_Proximity'])

# --- 4. CÁLCULO DA PONTUAÇÃO FINAL E EXIBIÇÃO ---
score_columns = [col for col in results_df.columns if col.startswith('Score')]
results_df['Pontuação Final'] = results_df[score_columns].sum(axis=1)

# Preparação e ordenação da tabela final
results_df_final = results_df.reset_index()
results_df_final[results_df_final.columns.difference(score_columns + ['Pontuação Final', 'tecnica'])] = results_df_final[results_df_final.columns.difference(score_columns + ['Pontuação Final', 'tecnica'])].round(4)

score_column_map = {
    f'Score Acurácia': 'Pt. AC', f'Score AUC': 'Pt. AUC', f'Score F1': 'Pt. F1', 
    f'Score TPR': 'Pt. TPR', f'Score TNR': 'Pt. TNR', f'Score FPR': 'Pt. FPR', 
    f'Score FNR': 'Pt. FNR', f'Score Predictive Parity Ratio': 'Pt. PPR'
}

results_df_output = results_df_final.rename(columns=score_column_map)
results_df_output = results_df_output.sort_values(by='Pontuação Final', ascending=False)


print("\nMédias Calculadas\n")
print(results_df_output[['tecnica', 'Acurácia', 'AUC', 'F1', 'TPR', 'TNR', 'FPR', 'FNR', 'Predictive Parity Ratio']].to_markdown(index=False, numalign="center", stralign="left"))

print("\nPontuação Individual e Final\n")
print(results_df_output.drop(columns=['Acurácia', 'AUC', 'F1', 'TPR', 'TNR', 'FPR', 'FNR', 'Predictive Parity Ratio']).to_markdown(index=False, numalign="center", stralign="left"))