"""
Código para mesclar os resultados do Vinícius em um único arquivo xlsx
A conversão para usar os mesmos termos nas colunas também é feitas aqui.
"""

import pandas as pd
import re
import os

# --- CONFIGURAÇÕES ---
bases = [1, 2, 3, 4]
sensitives = ['age', 'sexo']
output_name = 'resultado_vinicius.xlsx'

# Definição rigorosa das colunas (Padrão Global/Gabriel)
cols_completos = ['dataset', 'modelo', 'tecnica', 'cenario', 'smote', 'tempo_cpu', 'acuracia_teste', 'recall_teste', 'precisao_teste', 'f1_teste', 'auc_teste', 'ks_teste', 'classification_report_teste', 'confusion_matrix_teste']
cols_fairness = ['dataset', 'cenario', 'smote', 'modelo', 'tecnica', 'feature', 'group', 'count', 'accuracy', 'recall', 'precision', 'f1', 'confusion_matrix', 'true_positive_rate', 'true_negative_rate', 'false_positive_rate', 'false_negative_rate', 'selection_rate']
cols_extended = ['dataset', 'cenario', 'smote', 'modelo', 'tecnica', 'feature', 'demographic_parity_difference', 'demographic_parity_ratio', 'equalized_odds_difference', 'equalized_odds_ratio', 'equal_opportunity_difference', 'equal_opportunity_ratio']
cols_params = ['dataset', 'modelo', 'tecnica', 'cenario', 'smote', 'melhores_parametros']

data_completos, data_fairness, data_extended, data_params = [], [], [], []

for b in bases:
    for s in sensitives:
        file = f'base{b}_{s}_resultados_metricas.csv'
        if not os.path.exists(file): continue
        
        df = pd.read_csv(file)
        
        for _, row in df.iterrows():
            modelo_raw = str(row['modelo'])
            
            # 1. Filtro: Remove Logistic Regression puro, mantém Ridge (L2)
            if modelo_raw.strip() == "Logistic Regression":
                continue

            # 2. Definição do SMOTE e Dataset
            smote_val = 'smote_simples' if b in [1, 2] else 'original'
            dataset_name = f"dataset_{b}_{smote_val}_com_sensitive_{s}"
            
            # 3. Mapeamento de Técnica (Incluindo PostProcessing)
            if any(x in modelo_raw for x in ["ExpGrad", "InProcessing"]):
                tecnica = "exponentiated_gradient_eo"
            elif any(x in modelo_raw for x in ["TO (EO)", "PostProcessing"]):
                tecnica = "threshold_optimization_eo"
            else:
                tecnica = "nenhuma"
            
            # 4. Limpeza do Nome do Modelo (Ridge -> regressao_logistica)
            modelo_limpo = modelo_raw.split(' +')[0].lower()
            if 'ridge' in modelo_limpo or 'logistic' in modelo_limpo:
                modelo_limpo = 'regressao_logistica'
            modelo_limpo = modelo_limpo.replace(' ', '_')

            # Estrutura comum
            common = {
                'dataset': dataset_name, 'modelo': modelo_limpo, 'tecnica': tecnica,
                'cenario': 'Com Variáveis Sensíveis', 'smote': smote_val
            }
            
            # --- Preenchimento das Listas ---
            
            # Aba: resultados_completos
            res_comp = common.copy()
            res_comp.update({
                'tempo_cpu': '-', 'acuracia_teste': row.get('acuracia_teste'),
                'recall_teste': row.get('recall_teste'), 'precisao_teste': row.get('precisao_teste'),
                'f1_teste': row.get('f1_teste'), 'auc_teste': row.get('auc_teste')
            })
            data_completos.append(res_comp)
            
            # Aba: fairness_results (Pode estar vazia no CSV original do Vinícius, mas criamos a entrada)
            res_fair = common.copy()
            res_fair.update({'feature': f"sensitive_{s}"})
            data_fairness.append(res_fair)
            
            # Aba: extended_fairness_results
            res_ext = common.copy()
            res_ext.update({
                'feature': f"sensitive_{s}",
                'demographic_parity_difference': row.get('demographic_parity_difference'),
                'demographic_parity_ratio': row.get('demographic_parity_ratio'),
                'equalized_odds_difference': row.get('equalized_odds_difference'),
                'equal_opportunity_difference': row.get('equal_opportunity_difference')
            })
            data_extended.append(res_ext)
            
            # Aba: parametros_completos
            res_param = common.copy()
            res_param['melhores_parametros'] = row.get('melhores_parametros')
            data_params.append(res_param)

# Geração do Excel com Reindexação (Garante colunas extras vazias)
with pd.ExcelWriter(output_name, engine='xlsxwriter') as writer:
    pd.DataFrame(data_completos).reindex(columns=cols_completos).to_excel(writer, sheet_name='resultados_completos', index=False)
    pd.DataFrame(data_fairness).reindex(columns=cols_fairness).to_excel(writer, sheet_name='fairness_results', index=False)
    pd.DataFrame(data_extended).reindex(columns=cols_extended).to_excel(writer, sheet_name='extended_fairness_results', index=False)
    pd.DataFrame(data_params).reindex(columns=cols_params).to_excel(writer, sheet_name='parametros_completos', index=False)

print(f"Sucesso! '{output_name}' gerado com 4 abas e colunas padronizadas.")