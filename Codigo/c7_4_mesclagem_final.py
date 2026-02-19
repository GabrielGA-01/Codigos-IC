import pandas as pd

arquivos = {
    'Gabriel': 'resultado_global_19_02_2026_11_47.xlsx',
    'Vinicius': 'resultado_vinicius.xlsx',
    'Julia': 'resultado_julia.xlsx'
}

ABAS = ['resultados_completos', 'fairness_results', 'extended_fairness_results', 'parametros_completos']

# --- PARTE 1: MESCLAGEM DOS RESULTADOS ---
nome_final = 'Mesclagem_Resultados_Final.xlsx'
writer_final = pd.ExcelWriter(nome_final, engine='xlsxwriter')

for aba in ABAS:
    dfs = []
    for nome, arq in arquivos.items():
        df_temp = pd.read_excel(arq, sheet_name=aba)
        
        # Ajuste específico para Gabriel: Renomear técnica de threshold
        if nome == 'Gabriel':
            df_temp['tecnica'] = df_temp['tecnica'].replace('threshold_optimization', 'threshold_optimization_dp')
        
        # Regra: Remover 'nenhuma' para Gabriel e Vinícius
        if nome in ['Gabriel', 'Vinicius']:
            df_temp = df_temp[df_temp['tecnica'] != 'nenhuma']
            
        dfs.append(df_temp)
    
    df_mesclado = pd.concat(dfs, ignore_index=True, sort=False)
    df_mesclado.to_excel(writer_final, sheet_name=aba, index=False)

writer_final.close()
print(f"Sucesso: Arquivo '{nome_final}' salvo. (Gabriel atualizado para threshold_optimization_dp)")


# --- PARTE 2: COMPARAÇÃO ENTRE OS CASOS DE NENHUMA TÉCNICA ---
nome_comp = 'Mesclagem_Comparativa_Baselines.xlsx'
writer_comp = pd.ExcelWriter(nome_comp, engine='xlsxwriter')

for aba in ABAS:
    dfs_baseline = []
    for nome, arq in arquivos.items():
        df_temp = pd.read_excel(arq, sheet_name=aba)
        
        # Ajuste específico para Gabriel (mesmo sendo baseline, mantemos a consistência de nomes se houver)
        if nome == 'Gabriel':
            df_temp['tecnica'] = df_temp['tecnica'].replace('threshold_optimization', 'threshold_optimization_dp')
        
        # Filtrar apenas técnica 'nenhuma'
        df_baseline = df_temp[df_temp['tecnica'] == 'nenhuma'].copy()
        
        # Adicionar coluna do pesquisador no início
        df_baseline.insert(0, 'pesquisador', nome)
        
        dfs_baseline.append(df_baseline)
    
    df_final_comp = pd.concat(dfs_baseline, ignore_index=True, sort=False)
    df_final_comp.to_excel(writer_comp, sheet_name=aba, index=False)

writer_comp.close()
print(f"Sucesso: Arquivo '{nome_comp}' salvo com baselines.")

print("\n--- Todos os arquivos foram atualizados e salvos com sucesso! ---")