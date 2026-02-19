"""
Código para mesclar especificamente as planilhas da base1 do Vinícius.
"""

import pandas as pd

# --- CONFIGURAÇÃO ---
# Altere aqui para 'age' ou 'sexo'
sensivel = 'sexo' 

# Construção dinâmica dos nomes dos arquivos
file_perf = f'base1_{sensivel}_resultados_performance.csv'
file_just = f'base1_{sensivel}_resultados_justica.csv'
file_output = f'base1_{sensivel}_resultados_metricas.csv'

print(f"Iniciando mesclagem para a variável sensível: {sensivel}...")

# 1. Carregar os arquivos
try:
    df_perf = pd.read_csv(file_perf)
    df_just = pd.read_csv(file_just)
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Verifique se {file_perf} e {file_just} existem.")
    exit()

# 2. Mesclar usando 'modelo' como chave
# Usamos 'outer' ou 'inner'. Se os modelos forem idênticos em ambos, 'inner' é o ideal.
df_base1_unificada = pd.merge(df_perf, df_just, on='modelo', how='inner')

# 3. Definir a ordem das colunas baseada no padrão da Base 2
colunas_ordem = [
    'modelo', 'f1_macro_cv', 'acuracia_teste', 'recall_teste', 
    'precisao_teste', 'f1_teste', 'melhores_parametros',
    'demographic_parity_difference', 'demographic_parity_ratio',
    'equalized_odds_difference', 'equal_opportunity_difference'
]

# 4. Filtrar e Reordenar (Garante que só pegamos o necessário e na ordem certa)
# O .reindex garante que o código não quebre se uma coluna faltar
df_final = df_base1_unificada.reindex(columns=colunas_ordem)

# 5. Salvar o resultado
df_final.to_csv(file_output, index=False)

print(f"✓ Mesclagem concluída com sucesso!")
print(f"✓ Arquivo gerado: {file_output}")