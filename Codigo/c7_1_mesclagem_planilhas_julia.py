"""
Código para mesclar os resultados da Julia em um único arquivo xlsx
A conversão para usar os mesmos termos nas colunas também é feitas aqui.
"""

import pandas as pd
import re
import os

# --- CONFIGURAÇÕES ---
BASES = range(1, 5)
VARIAVEIS = ['age', 'sexo']
METODOS = ['SMOTE', 'original']
ABAS = ['resultados_completos', 'fairness_results', 'extended_fairness_results', 'parametros_completos']

OUTPUT_NAME = 'resultado_julia.xlsx'

# --- FUNÇÕES DE APOIO ---

def limpar_matriz(valor):
    if pd.isna(valor): return valor
    # Remove quebras de linha e espaços extras nas extremidades
    limpo = str(valor).replace('\n', ' ').strip()
    
    # Remove espaços duplos ou triplos internos
    limpo = re.sub(r'\s+', ' ', limpo)
    
    # REGEX: Procura um dígito seguido de espaço e outro dígito e insere a vírgula
    # Ex: "[[100 20] [5 30]]" -> "[[100, 20], [5, 30]]"
    limpo = re.sub(r'(\d+)\s+(?=\d+)', r'\1, ', limpo)
    
    # Garante que fechamento de colchete seguido de abertura também tenha vírgula
    # Ex: "[10, 20] [5, 30]" -> "[10, 20], [5, 30]"
    limpo = re.sub(r'\]\s+\[', r'], [', limpo)
    
    return limpo

def padronizar_dados_julia(df, aba_nome):
    """Aplica as regras de negócio para converter os termos da Julia para o padrão Global"""
    
    # 1. Ajuste de decimais (vírgula para ponto)
    cols_obj = df.select_dtypes(include=['object']).columns
    for col in cols_obj:
        if df[col].astype(str).str.contains(r'^\d+,\d+$').any():
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float, errors='ignore')

    # 2. Mapeamento de Modelo e Técnica
    def processar_modelo(row):
        original = str(row['modelo'])
        
        # Identificação da Técnica conforme solicitado
        if "ExpGrad" in original or "EG" in original or "DemographicParity" in original:
            tecnica = 'exponentiated_gradient_dp' # ou _dp se preferir manter o padrão exato da Julia
        else:
            tecnica = 'nenhuma'
        
        # Padronização do nome do Modelo
        modelo_limpo = original.split(' +')[0].lower()
        if 'ridge' in modelo_limpo or 'logistic' in modelo_limpo:
            modelo_limpo = 'regressao_logistica'
        elif 'perceptron' in modelo_limpo:
            modelo_limpo = 'perceptron'
        
        return pd.Series([modelo_limpo.replace(' ', '_'), tecnica])

    df[['modelo', 'tecnica']] = df.apply(processar_modelo, axis=1)

    # 3. Regra de SMOTE e Nome do Dataset
    def formatar_dataset_e_smote(row):
        # Extrai número da base do campo base_folder (ex: 'base1_age' -> 1)
        base_num = int(re.findall(r'\d+', str(row['base_folder']))[0])
        # Regra: Bases 1 e 2 são smote_simples, demais original
        smote_label = 'smote_simples' if base_num in [1, 2] else 'original'
        # Extrai a variável sensível (age/sexo)
        feature = str(row['base_folder']).split('_')[-1]
        
        dataset_name = f"dataset_{base_num}_{smote_label}_com_sensitive_{feature}"
        return pd.Series([smote_label, dataset_name])

    df[['smote', 'dataset']] = df.apply(formatar_dataset_e_smote, axis=1)

    # 4. Ajustes de Colunas Extras e Matrizes
    if 'tempo_cpu' not in df.columns: df['tempo_cpu'] = '-'
    df['cenario'] = 'Com Variáveis Sensíveis'
    
    col_matriz = 'confusion_matrix_teste' if 'confusion_matrix_teste' in df.columns else 'confusion_matrix'
    if col_matriz in df.columns:
        df[col_matriz] = df[col_matriz].apply(limpar_matriz)
        
    return df

# --- PROCESSAMENTO ---

writer = pd.ExcelWriter(OUTPUT_NAME, engine='xlsxwriter')

for aba in ABAS:
    print(f"Consolidando aba: {aba}...")
    acumulado_aba = []
    
    for n in BASES:
        for var in VARIAVEIS:
            for met in METODOS:
                nome_arq = f"base{n}_{var}_{met}.xlsx"
                if os.path.exists(nome_arq):
                    try:
                        df_temp = pd.read_excel(nome_arq, sheet_name=aba)
                        acumulado_aba.append(df_temp)
                    except Exception as e:
                        print(f" Erro ao ler {aba} em {nome_arq}: {e}")
    
    if acumulado_aba:
        df_unificado = pd.concat(acumulado_aba, ignore_index=True)
        df_padronizado = padronizar_dados_julia(df_unificado, aba)
        
        # Define as colunas finais desejadas para cada aba (Padrão Global)
        if aba == 'resultados_completos':
            cols = ['dataset', 'modelo', 'tecnica', 'cenario', 'smote', 'tempo_cpu', 'acuracia_teste', 'recall_teste', 'precisao_teste', 'f1_teste', 'auc_teste', 'ks_teste', 'classification_report_teste', 'confusion_matrix_teste']
        elif aba == 'extended_fairness_results':
            cols = ['dataset', 'cenario', 'smote', 'modelo', 'tecnica', 'feature', 'demographic_parity_difference', 'demographic_parity_ratio', 'equalized_odds_difference', 'equalized_odds_ratio', 'equal_opportunity_difference', 'equal_opportunity_ratio']
        elif aba == 'parametros_completos':
            cols = ['dataset', 'modelo', 'tecnica', 'cenario', 'smote', 'melhores_parametros']
        else: # fairness_results
            cols = ['dataset', 'cenario', 'smote', 'modelo', 'tecnica', 'feature', 'group', 'count', 'accuracy', 'recall', 'precision', 'f1', 'confusion_matrix']

        # Reordenar e salvar
        df_final = df_padronizado.reindex(columns=cols)
        df_final.to_excel(writer, sheet_name=aba, index=False)

writer.close()
print(f"\nSucesso! Arquivo '{OUTPUT_NAME}' gerado com dados da Julia padronizados.")