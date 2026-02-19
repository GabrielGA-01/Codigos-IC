import pandas as pd
import numpy as np
import os
import glob
from sklearn.linear_model import LinearRegression
from tabulate import tabulate  # Para formatação de tabelas no console

# --- CONFIGURAÇÕES BASEADAS NO ARTIGO ---
WEIGHT_W = 0.5  # Peso equilibrado (1-w) * Acurácia - (w) * Trade-off
CAMINHO_RESULTADOS = "./Resultados"

def clean_and_convert(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_pareto_front(df, score_col, fairness_prox_col):
    """Identifica modelos que não são dominados por nenhum outro (Fronteira de Pareto)."""
    pareto_front = []
    for i, row_a in df.iterrows():
        is_dominated = False
        for j, row_b in df.iterrows():
            # Dominação: B é melhor que A em uma métrica e não pior na outra
            if (row_b[score_col] > row_a[score_col] and row_b[fairness_prox_col] >= row_a[fairness_prox_col]) or \
               (row_b[fairness_prox_col] > row_a[fairness_prox_col] and row_b[score_col] >= row_a[score_col]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(row_a)
    return pd.DataFrame(pareto_front)

def processar_sp_ranking(nome_arquivo):
    print(f"\n" + "="*90)
    print(f"ANÁLISE MULTI-OBJETIVO: ESTRATÉGIA S_p (ARTIGO E PLURIBUS UNUM)")
    print(f"="*90)

    try:
        # 1. CARGA DE DADOS
        df_perf = pd.read_excel(nome_arquivo, sheet_name="resultados_completos")
        df_fair = pd.read_excel(nome_arquivo, sheet_name="extended_fairness_results")
        
        # Merge e renomeação para facilitar
        df = df_perf.merge(df_fair, on=['dataset', 'modelo', 'tecnica'], how='inner')
        df = df.rename(columns={'f1_teste': 'F1', 'equalized_odds_difference': 'EOD'})
        df = df[['dataset', 'modelo', 'tecnica', 'F1', 'EOD']].dropna()
        
        # Converter para proximidade (quanto maior, mais justo)
        df['Justica'] = 1 - np.abs(df['EOD'])
        
        resultados_finais = []

        # 2. PROCESSAMENTO POR DATASET
        for dataset_name, group in df.groupby('dataset'):
            # --- FILTRAGEM (REQUISITOS MÍNIMOS) ---
            # Pegamos o desempenho do modelo "Original" (Nenhuma técnica) como baseline
            baseline = group[group['tecnica'].str.upper() == 'NENHUMA']
            
            if not baseline.empty:
                f1_min = baseline['F1'].mean() * 0.9  # Aceitamos perder até 10% de F1 do original
                eod_max = baseline['EOD'].mean() * 1.1 # Aceitamos ser até 10% menos justo que o original
            else:
                f1_min = 0.50 # Fallback
                eod_max = 0.20 # Fallback

            # Aplicar filtro de viabilidade
            group_filtered = group[(group['F1'] >= f1_min) & (group['EOD'] <= eod_max)].copy()
            
            if len(group_filtered) < 2:
                # Se o filtro for rigoroso demais, usamos o grupo original para não quebrar o cálculo
                group_filtered = group.copy()

            # 3. CÁLCULO DA FRONTEIRA DE PARETO
            pareto = get_pareto_front(group_filtered, 'F1', 'Justica')
            
            # 4. TREINAR FUNÇÃO DE TRADE-OFF (Regressão Linear nos pontos de Pareto)
            # Objetivo: Prever o F1 ideal para um nível de Justiça
            reg = LinearRegression()
            X_p = pareto[['Justica']].values
            y_p = pareto['F1'].values
            
            if len(pareto) > 1:
                reg.fit(X_p, y_p)
            else:
                # Caso só haja um ponto Pareto, treinamos em todos os modelos filtrados
                reg.fit(group_filtered[['Justica']].values, group_filtered['F1'].values)

            # 5. CALCULAR S_p SCORE PARA TODOS
            for _, row in group.iterrows():
                f1_real = row['F1']
                justica_real = row['Justica']
                
                # F1 Esperado segundo a fronteira de Pareto
                f1_esperado = reg.predict([[justica_real]])[0]
                
                # S_p = (1-w)*F1_real - (w)*F1_esperado
                sp_score = (1 - WEIGHT_W) * f1_real - (WEIGHT_W * f1_esperado)
                
                # Tag de viabilidade
                viavel = "S" if (row['F1'] >= f1_min and row['EOD'] <= eod_max) else "N"
                
                resultados_finais.append({
                    'Dataset': dataset_name,
                    'Modelo': row['modelo'],
                    'Técnica': row['tecnica'],
                    'F1': f1_real,
                    'EOD': row['EOD'],
                    'S_p_Score': sp_score,
                    'Viável': viavel
                })

        # 6. EXIBIÇÃO DOS RANKINGS
        df_ranking = pd.DataFrame(resultados_finais)
        
        for dataset in df_ranking['Dataset'].unique():
            print(f"\n>>> DATASET: {dataset}")
            subset = df_ranking[df_ranking['Dataset'] == dataset].copy()
            
            # Ordenar por S_p Score (Eficiência)
            subset = subset.sort_values(by='S_p_Score', ascending=False)
            
            # Formatar para exibição
            subset['F1'] = subset['F1'].map('{:.4f}'.format)
            subset['EOD'] = subset['EOD'].map('{:.4f}'.format)
            subset['S_p_Score'] = subset['S_p_Score'].map('{:.6f}'.format)
            
            print(tabulate(subset.drop(columns='Dataset'), headers='keys', tablefmt='psql', showindex=False))
            
            vencedor = subset.iloc[0]
            print(f"ESTRATÉGIA RECOMENDADA: {vencedor['Técnica']} com {vencedor['Modelo']}")
            print("-" * 90)

    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")

# --- EXECUÇÃO ---
if __name__ == "__main__":
    # Localizar arquivo mais recente
    arquivo = f'{CAMINHO_RESULTADOS}/Mesclagem_Resultados_Final.xlsx'
    processar_sp_ranking(arquivo)