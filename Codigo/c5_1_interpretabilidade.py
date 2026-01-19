"""
Este arquivo coordena a execução de múltiplas técnicas de interpretabilidade global.
Ele percorre os resultados dos modelos treinados e chama as funções para calcular 
SHAP, Permutation Importance e LIME.
"""

import pandas as pd
import os
import time
from c5_2_shap import importancia_shap
from c5_3_permutation_importance import importancia_permutacao
from c5_4_lime import importancia_lime

def gerar_interpretabilidade(dados, caminho_resultado):
    """
    Gera as métricas de interpretabilidade (SHAP, Permutation, LIME) para todos os modelos
    presentes no dicionário de dados (apenas para 'sem_tecnica' para evitar duplicação).
    """
    
    lista_shap = []
    lista_permutation = []
    lista_lime = []

    print("Iniciando cálculo de interpretabilidade...")

    # Iterar sobre a estrutura do dicionário
    for label_dataset, resto_dataset in dados.items():
        
        if not isinstance(resto_dataset, dict): 
            continue 
            
        modelos = {k: v for k, v in resto_dataset.items() if k != 'smote'}

        for modelo_nome, tecnicas in modelos.items():
            
            for tecnica_nome, dados_tecnica in tecnicas.items():
                
                # Pular caso não seja um dicionário de resultados válido
                if not isinstance(dados_tecnica, dict):
                    continue

                # Verificar se existe a chave de interpretabilidade
                if 'geral' in dados_tecnica and 'interpretabilidade' in dados_tecnica['geral']:
                    info_interp = dados_tecnica['geral']['interpretabilidade']
                    
                    pipeline = info_interp['pipeline']
                    nome_modelo = info_interp['nome_modelo']
                    nome_dataset = info_interp['nome_dataset']
                    cenario = info_interp['cenario']
                    X_train = info_interp['X_train']
                    X_test = info_interp['X_test']
                    y_test = info_interp['y_test']

                    # Adicionar nome da técnica para identificação no Excel
                    nome_identificacao = f"{nome_modelo} ({tecnica_nome})"

                    print(f"Processando interpretabilidade para: {nome_dataset}")

                    # 1. SHAP
                    try:
                        start_shap = time.time()
                        res_shap = importancia_shap(pipeline, nome_modelo, nome_dataset, cenario, X_test)
                        end_shap = time.time()
                        print(f"  > SHAP concluído em {end_shap - start_shap:.2f}s")
                        
                        # Adicionar identificador da técnica no resultado
                        for item in res_shap:
                            item['tecnica'] = tecnica_nome
                        lista_shap.extend(res_shap)
                    except Exception as e:
                        print(f"Erro ao calcular SHAP para {nome_identificacao}: {e}")

                    # 2. Permutation Importance
                    try:
                        start_perm = time.time()
                        res_perm = importancia_permutacao(pipeline, nome_modelo, nome_dataset, cenario, X_test, y_test)
                        end_perm = time.time()
                        print(f"  > Permutation Importance concluída em {end_perm - start_perm:.2f}s")

                        for item in res_perm:
                            item['tecnica'] = tecnica_nome
                        lista_permutation.extend(res_perm)
                    except Exception as e:
                        print(f"Erro ao calcular Permutation Importance para {nome_identificacao}: {e}")

                    # 3. LIME
                    try:
                        start_lime = time.time()
                        res_lime = importancia_lime(pipeline, nome_modelo, nome_dataset, cenario, X_train, X_test)
                        end_lime = time.time()
                        print(f"  > LIME concluído em {end_lime - start_lime:.2f}s")

                        for item in res_lime:
                            item['tecnica'] = tecnica_nome
                        lista_lime.extend(res_lime)
                    except Exception as e:
                        print(f"Erro ao calcular LIME para {nome_identificacao}: {e}")

    # Criar DataFrames
    df_shap = pd.DataFrame(lista_shap)
    df_permutation = pd.DataFrame(lista_permutation)
    df_lime = pd.DataFrame(lista_lime)

    # Salvar em Excel
    nome_arquivo = f'{caminho_resultado}/interpretabilidade_global_{time.strftime("%d_%m_%Y_%H_%M", time.localtime())}.xlsx'

    try:
        if os.path.isfile(nome_arquivo):
            print("Atenção! Há uma planilha de interpretabilidade com o mesmo nome!")
            print("Salvando com o nome _aux")
            nome_arquivo = nome_arquivo.replace("_global", "_global_aux")

        with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
            if not df_shap.empty:
                df_shap.to_excel(writer, sheet_name='shap', index=False)
            if not df_permutation.empty:
                df_permutation.to_excel(writer, sheet_name='permutation', index=False)
            if not df_lime.empty:
                df_lime.to_excel(writer, sheet_name='lime', index=False)

        print(f"Arquivo de interpretabilidade salvo em {nome_arquivo}")
        

    except Exception as e:
        print(f"Erro ao salvar o arquivo Excel de interpretabilidade: {e}")

def gerar_interpretabilidade_especifica(dados, nome_dataset, nome_modelo, nome_tecnica):
    """
    Gera as métricas de interpretabilidade (SHAP, Permutation, LIME) para UM CASO ESPECÍFICO.
    Retorna um dicionário com os resultados, sem salvar em arquivo.
    """
    resultados_interp = {
        'shap': None,
        'permutation': None,
        'lime': None
    }

    print(f"Iniciando cálculo de interpretabilidade específico para: {nome_dataset} | {nome_modelo} | {nome_tecnica}")

    try:
        if nome_dataset not in dados:
            raise ValueError(f"Dataset '{nome_dataset}' não encontrado.")
        if nome_modelo not in dados[nome_dataset]:
             raise ValueError(f"Modelo '{nome_modelo}' não encontrado.")
        if nome_tecnica not in dados[nome_dataset][nome_modelo]:
             raise ValueError(f"Técnica '{nome_tecnica}' não encontrada.")

        dados_tecnica = dados[nome_dataset][nome_modelo][nome_tecnica]

        if 'geral' in dados_tecnica and 'interpretabilidade' in dados_tecnica['geral']:
            info_interp = dados_tecnica['geral']['interpretabilidade']
            
            pipeline = info_interp['pipeline']
            nome_modelo_real = info_interp['nome_modelo']
            cenario = info_interp['cenario']
            X_train = info_interp['X_train']
            X_test = info_interp['X_test']
            y_test = info_interp['y_test']

            nome_identificacao = f"{nome_modelo} ({nome_tecnica})"

            # 1. SHAP
            try:
                start_shap = time.time()
                res_shap = importancia_shap(pipeline, nome_modelo_real, nome_dataset, cenario, X_test)
                end_shap = time.time()
                print(f"  > SHAP concluído em {end_shap - start_shap:.2f}s")
                
                # Adicionar identificador
                for item in res_shap:
                    item['tecnica'] = nome_tecnica
                resultados_interp['shap'] = res_shap
            except Exception as e:
                print(f"Erro ao calcular SHAP para {nome_identificacao}: {e}")

            # 2. Permutation Importance
            try:
                start_perm = time.time()
                res_perm = importancia_permutacao(pipeline, nome_modelo_real, nome_dataset, cenario, X_test, y_test)
                end_perm = time.time()
                print(f"  > Permutation Importance concluída em {end_perm - start_perm:.2f}s")

                for item in res_perm:
                    item['tecnica'] = nome_tecnica
                resultados_interp['permutation'] = res_perm
            except Exception as e:
                print(f"Erro ao calcular Permutation Importance para {nome_identificacao}: {e}")

            # 3. LIME
            try:
                start_lime = time.time()
                res_lime = importancia_lime(pipeline, nome_modelo_real, nome_dataset, cenario, X_train, X_test)
                end_lime = time.time()
                print(f"  > LIME concluído em {end_lime - start_lime:.2f}s")

                for item in res_lime:
                    item['tecnica'] = nome_tecnica
                resultados_interp['lime'] = res_lime
            except Exception as e:
                print(f"Erro ao calcular LIME para {nome_identificacao}: {e}")
        else:
            print(f"Dados de interpretabilidade ('pipeline', 'X_train', etc.) não encontrados para este caso.")

    except Exception as e:
        print(f"Erro geral na interpretabilidade específica: {e}")

    return resultados_interp
