"""
Este arquivo contém as funções para persistência dos resultados do projeto.
Inclui o salvamento do dicionário de métricas em formato joblib e a geração de 
planilhas Excel detalhadas com métricas de desempenho e justiça.
"""

import joblib
import pandas as pd
import os
import time
import re

def salvar_dicionario(resultado, caminho_resultado, sucesso):
    """
    Salva o dicionário de resultados em um arquivo .joblib.
    
    Parâmetros:
    - resultado: O dicionário contendo todas as métricas acumuladas.
    - caminho_resultado: Diretório onde o arquivo será salvo.
    - sucesso: Booleano para definir o prefixo do nome (global ou parcial).
    """
    nome = ''
    if sucesso:
        nome = 'global'
    else:
        nome = 'parcial'

    caminho = f'{caminho_resultado}/resultado_{nome}_dict_{time.strftime("%d_%m_%Y_%H_%M", time.localtime())}.joblib'
    if os.path.isfile(caminho):
        print("Atenção! Há um dicionário com o mesmo nome!")
        print("Salvando com o nome _aux")
        caminho = caminho.replace("_dict", "_dict_aux")
    joblib.dump(resultado, caminho)
    print(f"Dicionário com o resultado {nome} salvo em {caminho}")

# Função auxiliar para extrair a variável sensível (feature)
def extrair_feature(label_dataset):
    """Extrai o nome da feature sensível do label_dataset."""
    if 'sexo' in label_dataset.lower():
        return 'sensitive_sexo'
    elif 'age' in label_dataset.lower():
        return 'sensitive_age'
    return None

# Função para calcular métricas de taxa a partir da matriz de confusão (Mantido do código anterior)
def calcular_metricas_de_matriz(matriz_str):
    """Calcula TPR, TNR, FPR, FNR, Selection Rate e Count a partir da matriz de confusão string."""
    try:
        # Extrai a matriz (assumindo formato '[[TN, FP], [FN, TP]]')
        # Também lida com a matriz vinda como string de numpy (ex: "[187  13]\n [ 64  23]]")
        matriz_limpa = re.findall(r'[\d.]+', matriz_str)
        matriz_valores = [float(v) for v in matriz_limpa]
        
        if len(matriz_valores) == 4:
            # Assumindo ordem: TN, FP, FN, TP ou TN, FP, FN, TP
            # Se vier de um .split(',') simples, é [TN, FP, FN, TP]
            # Se vier de numpy.array.__str__, a ordem pode ser diferente se houver quebra de linha. 
            # Vou manter a ordem padrão inferida: TN, FP, FN, TP
            TN, FP, FN, TP = matriz_valores[0], matriz_valores[1], matriz_valores[2], matriz_valores[3]
            
            soma_positivos_reais = TP + FN
            soma_negativos_reais = TN + FP
            count = TP + FP + TN + FN
            
            tpr = TP / soma_positivos_reais if soma_positivos_reais > 0 else 0
            tnr = TN / soma_negativos_reais if soma_negativos_reais > 0 else 0
            fpr = FP / soma_negativos_reais if soma_negativos_reais > 0 else 0
            fnr = FN / soma_positivos_reais if soma_positivos_reais > 0 else 0
            selection_rate = (TP + FP) / count if count > 0 else 0
            
            return tpr, tnr, fpr, fnr, selection_rate, count
        
        return [None] * 6
    except Exception:
        return [None] * 6
    
def gerar_planilha(dados, caminho_resultado):
    # Listas para os dados de cada planilha
    lista_resultados_completos = []
    lista_extended_fairness = []
    lista_fairness_results = [] 
    lista_parametros_completos = []

    # NOVO: Valor constante para a coluna 'cenario'
    CENARIO_VALOR = "Com Variáveis Sensíveis"

    # Mapeamento do grupo e nome da feature sensível
    MAP_SENSIBLE = {
        'sexo': {
            'privilegiado': {'group_id': 1, 'nome_feature': 'sensitive_sexo'}, 
            'desprivilegiado': {'group_id': 0, 'nome_feature': 'sensitive_sexo'}
        },
        'age': {
            'privilegiado': {'group_id': 1, 'nome_feature': 'sensitive_age'}, 
            'desprivilegiado': {'group_id': 0, 'nome_feature': 'sensitive_age'} 
        }
    }

    # Técnicas que não alteram o modelo, mas usam o modelo base (sem_tecnica)
    post_processing_tecnicas = [
        'threshold_optimization', 'calibration', 'reject_option_classification'
    ]

    # Iterar sobre a estrutura do dicionário
    for label_dataset, resto_dataset in dados.items():
        
        if not isinstance(resto_dataset, dict): 
            continue 
            
        smote_valor = resto_dataset.get('smote', 'Original (Inferido)') 
        
        # Extração de metadados
        dataset_base = label_dataset
        feature_sensivel_base = 'sexo' if 'sexo' in label_dataset.lower() else 'age'
        
        modelos = {k: v for k, v in resto_dataset.items() if k != 'smote'}

        for modelo, tecnicas in modelos.items():

            parametros_base_dict = tecnicas.get('sem_tecnica', {}).get('melhores_parametros')
            parametros_base_str = str(parametros_base_dict) if parametros_base_dict else None

            for tecnica_nome, metricas_tecnica in tecnicas.items():

                if tecnica_nome == 'label_modification':
                    continue

                metricas_gerais = metricas_tecnica.get('geral', {})
                relatorio_geral = metricas_gerais.get('relatorio_classificacao', {})
                if not relatorio_geral:
                    continue

                # Informações base
                info_base = {
                    'dataset': dataset_base, 
                    'smote': smote_valor, 
                    'modelo': modelo,
                    'tecnica': 'Nenhuma' if tecnica_nome == 'sem_tecnica' else tecnica_nome, # Adicionada a todas
                    'cenario': CENARIO_VALOR 
                }
                
                tempo_cpu = metricas_tecnica.get('tempo', {}).get('cpu')

                # =========================================================
                # 1. PLANILHA 'resultados_completos'
                # =========================================================
                linha_resultados = info_base.copy()
                linha_resultados['tempo_cpu'] = tempo_cpu 
                linha_resultados['acuracia_teste'] = relatorio_geral.get('accuracy')
                linha_resultados['auc_teste'] = metricas_gerais.get('ROC_AUC')
                linha_resultados['ks_teste'] = metricas_gerais.get('KS') 
                
                metricas_classe_1 = relatorio_geral.get('1', relatorio_geral.get('1.0', {}))
                linha_resultados['recall_teste'] = metricas_classe_1.get('recall')
                linha_resultados['precisao_teste'] = metricas_classe_1.get('precision')
                linha_resultados['f1_teste'] = metricas_classe_1.get('f1-score')
                linha_resultados['classification_report_teste'] = str(relatorio_geral)
                linha_resultados['confusion_matrix_teste'] = str(metricas_gerais.get('matriz_de_confusao'))
                
                lista_resultados_completos.append(linha_resultados)

                # =========================================================
                # 2. PLANILHA 'extended_fairness_results'
                # =========================================================
                linha_extended = info_base.copy()
                linha_extended['feature'] = extrair_feature(label_dataset)
                
                dp = metricas_gerais.get('demographic_parity', {})
                linha_extended['demographic_parity_difference'] = dp.get('demographic_parity_difference')
                linha_extended['demographic_parity_ratio'] = dp.get('demographic_parity_ratio')
                
                eo = metricas_gerais.get('equalized_odds', {})
                linha_extended['equalized_odds_difference'] = eo.get('equalized_odds_difference')
                linha_extended['equalized_odds_ratio'] = eo.get('equalized_odds_ratio')
                
                tpr = metricas_gerais.get('true_positive_rate', {})
                linha_extended['equal_opportunity_difference'] = tpr.get('true_positive_rate_difference')
                linha_extended['equal_opportunity_ratio'] = tpr.get('true_positive_rate_ratio')
                
                fpr_m = metricas_gerais.get('false_positive_rate', {})
                linha_extended['false_positive_rate_difference'] = fpr_m.get('false_positive_rate_difference')
                linha_extended['false_positive_rate_ratio'] = fpr_m.get('false_positive_rate_ratio')
                
                fnr_m = metricas_gerais.get('false_negative_rate', {})
                linha_extended['false_negative_rate_difference'] = fnr_m.get('false_negative_rate_difference')
                linha_extended['false_negative_rate_ratio'] = fnr_m.get('false_negative_rate_ratio')

                pp = metricas_gerais.get('predictive_parity', {})
                linha_extended['predictive_parity_difference'] = pp.get('predictive_parity_difference')
                linha_extended['predictive_parity_ratio'] = pp.get('predictive_parity_ratio')
                
                lista_extended_fairness.append(linha_extended)

                # =========================================================
                # 3. PLANILHA 'fairness_results'
                # =========================================================
                
                if feature_sensivel_base not in MAP_SENSIBLE:
                    continue

                grupos_status = {
                    'privilegiado': MAP_SENSIBLE[feature_sensivel_base]['privilegiado']['group_id'],
                    'desprivilegiado': MAP_SENSIBLE[feature_sensivel_base]['desprivilegiado']['group_id']
                }

                for status, grupo_valor in grupos_status.items():
                    metricas_grupo = metricas_tecnica.get(status) 
                    
                    if metricas_grupo:
                        relatorio_grupo = metricas_grupo.get('relatorio_classificacao', {})
                        matriz_str = str(metricas_grupo.get('matriz_de_confusao'))

                        tpr, tnr, fpr, fnr, selection_rate, count = calcular_metricas_de_matriz(matriz_str)

                        metricas_classe_1_grupo = relatorio_grupo.get('1', relatorio_grupo.get('1.0', {}))
                        
                        linha_fairness_results = {
                            'dataset': dataset_base, 
                            'cenario': CENARIO_VALOR, 
                            'smote': info_base['smote'],
                            'modelo': modelo, 
                            'tecnica': info_base['tecnica'], 
                            'feature': extrair_feature(label_dataset), 
                            'group': grupo_valor, 
                            'count': count, 
                            'accuracy': relatorio_grupo.get('accuracy'), 
                            'recall': metricas_classe_1_grupo.get('recall'), 
                            'precision': metricas_classe_1_grupo.get('precision'), 
                            'f1': metricas_classe_1_grupo.get('f1-score'), 
                            'confusion_matrix': matriz_str,
                            'true_positive_rate': tpr, 
                            'true_negative_rate': tnr, 
                            'false_positive_rate': fpr, 
                            'false_negative_rate': fnr, 
                            'selection_rate': selection_rate
                        }
                        lista_fairness_results.append(linha_fairness_results)

                # =========================================================
                # 4. PLANILHA 'parametros_completos'
                # =========================================================
                linha_parametro = info_base.copy()

                if tecnica_nome in post_processing_tecnicas:
                    linha_parametro['melhores_parametros'] = parametros_base_str
                else:
                    parametros_atuais = metricas_tecnica.get('melhores_parametros') 
                    linha_parametro['melhores_parametros'] = str(parametros_atuais) if parametros_atuais else None

                lista_parametros_completos.append(linha_parametro)


    # --- Criar os DataFrames ---
    df_resultados_completos = pd.DataFrame(lista_resultados_completos)
    df_extended_fairness = pd.DataFrame(lista_extended_fairness)
    df_fairness_results = pd.DataFrame(lista_fairness_results)
    df_parametros_completos = pd.DataFrame(lista_parametros_completos)


    # --- Definir a ordem das colunas e mapear nomes para o padrão Julia ---
    
    # 1. resultados_completos: INCLUÍDA 'tecnica' e REPOSICIONADA após 'modelo'
    ordem_resultados = [
        'dataset', 'modelo', 'tecnica', 'cenario', 'smote', 'tempo_cpu', 'acuracia_teste', 'recall_teste', 
        'precisao_teste', 'f1_teste', 'auc_teste', 'ks_teste', 'classification_report_teste', 
        'confusion_matrix_teste'
    ]

    # 2. extended_fairness_results: REPOSICIONADA 'tecnica' após 'modelo'
    ordem_extended = [
        'dataset', 'cenario', 'smote', 'modelo', 'tecnica', 'feature', 
        'demographic_parity_difference', 'demographic_parity_ratio', 
        'equalized_odds_difference', 'equalized_odds_ratio', 
        'equal_opportunity_difference', 'equal_opportunity_ratio', 
        'false_positive_rate_difference', 'false_positive_rate_ratio', 
        'false_negative_rate_difference', 'false_negative_rate_ratio', 
        'predictive_parity_difference', 'predictive_parity_ratio'
    ]

    # 3. fairness_results: REPOSICIONADA 'tecnica' após 'modelo'
    ordem_fairness_results = [
        'dataset', 'cenario', 'smote', 'modelo', 'tecnica', 'feature', 'group', 'count', 
        'accuracy', 'recall', 'precision', 'f1', 'confusion_matrix', 
        'true_positive_rate', 'true_negative_rate', 'false_positive_rate', 
        'false_negative_rate', 'selection_rate'
    ]

    # 4. parametros_completos: REPOSICIONADA 'tecnica' após 'modelo'
    ordem_parametros = [
        'dataset', 'modelo', 'tecnica', 'cenario', 'smote', 'melhores_parametros'
    ]

    # Aplicar reordenação
    df_resultados_completos = df_resultados_completos[[col for col in ordem_resultados if col in df_resultados_completos.columns]]
    df_extended_fairness = df_extended_fairness[[col for col in ordem_extended if col in df_extended_fairness.columns]]
    df_fairness_results = df_fairness_results[[col for col in ordem_fairness_results if col in df_fairness_results.columns]]
    df_parametros_completos = df_parametros_completos[[col for col in ordem_parametros if col in df_parametros_completos.columns]]


    # --- Salvar os DataFrames em um único arquivo Excel ---
    nome_arquivo = f'{caminho_resultado}/resultado_global_{time.strftime("%d_%m_%Y_%H_%M", time.localtime())}.xlsx'

    try:
        if os.path.isfile(nome_arquivo):
            print("Atenção! Há uma planilha com o mesmo nome!")
            print("Salvando com o nome _aux")
            nome_arquivo = nome_arquivo.replace("_global", "_global_aux")

        with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
            df_resultados_completos.to_excel(writer, sheet_name='resultados_completos', index=False)
            df_extended_fairness.to_excel(writer, sheet_name='extended_fairness_results', index=False)
            df_fairness_results.to_excel(writer, sheet_name='fairness_results', index=False)
            df_parametros_completos.to_excel(writer, sheet_name='parametros_completos', index=False)

        print(f"Arquivo Excel com {len(df_resultados_completos)} linhas de dados salvo em {nome_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo Excel: {e}")