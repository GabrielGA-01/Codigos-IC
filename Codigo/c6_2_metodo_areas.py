"""
Este arquivo implementa o cálculo da Métrica de Performance Multi-Critério (MCPM).
A métrica utiliza o cálculo da área de um triângulo formado por F1-Score, 
proximidade de Equalized Odds e proximidade de Demographic Parity.
Permite comparar técnicas e modelos de forma agregada.
"""

import pandas as pd
import numpy as np
import os
import glob

# Constante global
SIN_120 = np.sqrt(3) / 2

# Nomes das planilhas (sheets) a serem lidas
sheet_resultados_completos = "resultados_completos"
sheet_fairness_results = "fairness_results"
sheet_extended_fairness_results = "extended_fairness_results"

def clean_and_convert(df, cols):
    """Converte colunas para numérico, tratando possíveis erros de formatação."""
    for col in cols:
        # Usa errors='coerce' para transformar valores não numéricos (como strings ou NaN) em NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def criterio_ordenacao(nome):
    """Função de ordenação: Primeiro os que tem 'ORIGINAL', depois alfabético"""
    e_original = "ORIGINAL" in nome.upper()
    return (not e_original, nome)

def calcular_desempenho_tecnicas(nome_arquivo_excel):
    """
    Calcula a Métrica de Performance Multi-Critério (MCPM) agregando por dataset e TÉCNICA.
    Faz a média de todos os modelos dessa técnica.
    """
    try:
        print(f"\n--- INICIANDO ANÁLISE POR TÉCNICA (MÉDIA DOS MODELOS) ---")
        print(f"Lendo o arquivo Excel: {nome_arquivo_excel}")
        
        # Leitura da planilha de resultados de desempenho (F1-Score)
        df_comp = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_resultados_completos)
        df_comp = clean_and_convert(df_comp, ['f1_teste'])
        
        # Leitura da planilha de resultados de fairness estendida (EOR e DPR)
        df_ext_fair = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_extended_fairness_results)
        df_ext_fair = clean_and_convert(df_ext_fair, ['equalized_odds_difference', 'demographic_parity_difference'])
        
    except Exception as e:
        print(f"Erro ao ler ou processar o arquivo Excel: {e}")
        return

    # --- SELEÇÃO E FILTRAGEM ---
    cols_comp = ['dataset', 'tecnica', 'f1_teste']
    cols_fair = ['dataset', 'tecnica', 'equalized_odds_difference', 'demographic_parity_difference']

    df_comp = df_comp[cols_comp]
    df_ext_fair = df_ext_fair[cols_fair]

    # Join dos dataframes
    df_analise = df_comp.merge(df_ext_fair, on=['dataset', 'tecnica'], how='inner').dropna()

    # Renomear colunas
    df_analise = df_analise.rename(columns={
        'f1_teste': 'F1',
        'equalized_odds_difference': 'EOD',
        'demographic_parity_difference': 'DPD'
    })

    # --- CÁLCULO DA MÉDIA POR TÉCNICA DENTRO DE CADA DATASET ---
    df_analise = df_analise.groupby(['dataset', 'tecnica'])[['F1', 'EOD', 'DPD']].mean().reset_index()

    # --- CÁLCULO DA TÉCNICA (Métrica Composta) ---
    df_analise['EOD_Prox'] = 1 - np.abs(df_analise['EOD'])
    df_analise['DPD_Prox'] = 1 - np.abs(df_analise['DPD'])
    df_analise['EOD_Prox'] = df_analise['EOD_Prox'].clip(lower=0)
    df_analise['DPD_Prox'] = df_analise['DPD_Prox'].clip(lower=0)

    F1 = df_analise['F1']
    EOR_P = df_analise['EOD_Prox']
    DPR_P = df_analise['DPD_Prox']

    # Área Total
    df_analise['MCPM'] = (F1 * EOR_P * SIN_120) + (EOR_P * DPR_P * SIN_120) + (DPR_P * F1 * SIN_120)

    # --- EXIBIÇÃO DOS RESULTADOS ---
    datasets_unicos = df_analise['dataset'].unique()
    datasets_ordenados = sorted(datasets_unicos, key=criterio_ordenacao)

    print("\nMétrica de Performance Multi-Critério (MCPM) - Por Dataset (Média das Técnicas)\n")
    print(f"Métricas Utilizadas: F1-Score, Equalized Odds Diff (Prox. 0), Demographic Parity Diff (Prox. 0)")
    print("=" * 100)

    for dataset in datasets_ordenados:
        print(f"\nDATASET: {dataset}")
        print("-" * 100)
        
        df_subset = df_analise[df_analise['dataset'] == dataset].copy()
        df_subset = df_subset.sort_values(by='MCPM', ascending=False)
        
        cols_output = ['tecnica', 'F1', 'EOD_Prox', 'DPD_Prox', 'MCPM']
        df_subset = df_subset[cols_output]
        
        df_subset = df_subset.rename(columns={
            'EOD_Prox': 'EOD (1-|Diff|)',
            'DPD_Prox': 'DPD (1-|Diff|)'
        })
        
        df_subset = df_subset.round(4)
        
        print(df_subset.to_markdown(index=False, numalign="center", stralign="left"))
        print("-" * 100)

def comparar_desempenho_modelos(nome_arquivo_excel):
    """
    Calcula a Métrica de Performance Multi-Critério (MCPM) POR MODELO.
    Compara todos os modelos dentro de cada dataset, mostrando Modelo e Técnica.
    """
    try:
        print(f"\n--- INICIANDO COMPARAÇÃO DE MODELOS ---")
        
        # Leitura das planilhas - Agora incluindo 'modelo'
        df_comp = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_resultados_completos)
        df_comp = clean_and_convert(df_comp, ['f1_teste'])
        
        df_ext_fair = pd.read_excel(nome_arquivo_excel, sheet_name=sheet_extended_fairness_results)
        df_ext_fair = clean_and_convert(df_ext_fair, ['equalized_odds_difference', 'demographic_parity_difference'])
        
    except Exception as e:
        print(f"Erro ao ler arquivo para comparação de modelos: {e}")
        return

    # --- SELEÇÃO E FILTRAGEM ---
    # Incluindo 'modelo' nas colunas
    cols_comp = ['dataset', 'modelo', 'tecnica', 'f1_teste']
    cols_fair = ['dataset', 'modelo', 'tecnica', 'equalized_odds_difference', 'demographic_parity_difference']

    # Verificar se as colunas existem
    missing_cols = [c for c in cols_comp if c not in df_comp.columns]
    if missing_cols:
        print(f"Colunas faltando em resultados_completos: {missing_cols}")
        return
        
    missing_cols_fair = [c for c in cols_fair if c not in df_ext_fair.columns]
    if missing_cols_fair:
        print(f"Colunas faltando em extended_fairness_results: {missing_cols_fair}")
        return

    df_comp = df_comp[cols_comp]
    df_ext_fair = df_ext_fair[cols_fair]

    # Join dos dataframes usando dataset, modelo e tecnica
    df_analise = df_comp.merge(df_ext_fair, on=['dataset', 'modelo', 'tecnica'], how='inner').dropna()

    df_analise = df_analise.rename(columns={
        'f1_teste': 'F1',
        'equalized_odds_difference': 'EOD',
        'demographic_parity_difference': 'DPD'
    })

    # Se houver múltiplas entradas para o mesmo modelo/técnica/dataset (ex: k-fold), fazemos a média
    df_analise = df_analise.groupby(['dataset', 'modelo', 'tecnica'])[['F1', 'EOD', 'DPD']].mean().reset_index()

    # --- CÁLCULO DA TÉCNICA (Métrica Composta) ---
    df_analise['EOD_Prox'] = 1 - np.abs(df_analise['EOD'])
    df_analise['DPD_Prox'] = 1 - np.abs(df_analise['DPD'])
    df_analise['EOD_Prox'] = df_analise['EOD_Prox'].clip(lower=0)
    df_analise['DPD_Prox'] = df_analise['DPD_Prox'].clip(lower=0)

    F1 = df_analise['F1']
    EOR_P = df_analise['EOD_Prox']
    DPR_P = df_analise['DPD_Prox']

    # Área Total
    df_analise['MCPM'] = (F1 * EOR_P * SIN_120) + (EOR_P * DPR_P * SIN_120) + (DPR_P * F1 * SIN_120)

    # --- EXIBIÇÃO DOS RESULTADOS ---
    datasets_unicos = df_analise['dataset'].unique()
    datasets_ordenados = sorted(datasets_unicos, key=criterio_ordenacao)

    print("\n\n=== COMPARAÇÃO DE MODELOS (Ranking MCPM) ===")
    
    for dataset in datasets_ordenados:
        print(f"\nDATASET: {dataset}")
        print("-" * 120)
        
        df_subset = df_analise[df_analise['dataset'] == dataset].copy()
        df_subset = df_subset.sort_values(by='MCPM', ascending=False)
        
        cols_output = ['modelo', 'tecnica', 'F1', 'EOD_Prox', 'DPD_Prox', 'MCPM']
        df_subset = df_subset[cols_output]
        
        df_subset = df_subset.rename(columns={
            'EOD_Prox': 'EOD (1-|Diff|)',
            'DPD_Prox': 'DPD (1-|Diff|)'
        })
        
        df_subset = df_subset.round(4)
        
        print(df_subset.to_markdown(index=False, numalign="center", stralign="left"))
        print("-" * 120)

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    caminho_resultados = "./Resultados"
    # Tenta definir um padrão inicial, mas ajusta automaticamente
    nome_arquivo_excel = f"{caminho_resultados}/resultado_global_03_01_2026_19_50.xlsx"

    if not os.path.exists(nome_arquivo_excel):
        list_of_files = glob.glob(f'{caminho_resultados}/resultado_global_*.xlsx')
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"Arquivo padrão não encontrado. Usando o mais recente detectado: {latest_file}")
            nome_arquivo_excel = latest_file
        else:
            print(f"Erro: Nenhum arquivo 'resultado_global_*.xlsx' encontrado em {caminho_resultados}.")
            exit()
    
    # 1. Executa a análise original (agora em função)
    calcular_desempenho_tecnicas(nome_arquivo_excel)
    
    # 2. Executa a nova comparação por modelos
    comparar_desempenho_modelos(nome_arquivo_excel)
