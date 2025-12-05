from c0_1_configuracoes import(
    colunas_discretas1,
    colunas_discretas2,
    colunas_discretas3,
    colunas_discretas4,
    dados_sensiveis_age,
    dados_sensiveis_sexo
)

# Funções auxiliares
from c1_2_smote import enviesamento_smotenc_rotulo_binario as enviesamento_smotenc
from c1_3_preprocessor_automatizado import aplicar_pre_processador

import pandas as pd

def enviesar():
    # Enviesando os datasets
    datasets = {
        'df1':{
            'original': {
            'nome_banco': "DATASET 1 ORIGINAL",
            'data': pd.read_csv(f"Datasets/Processados/Dataset1.csv", sep=','),
            'colunas_discretas': colunas_discretas1,
            }
        },
        'df2':{
            'original': {
            'nome_banco': "DATASET 2 ORIGINAL",
            'data': pd.read_csv(f"Datasets/Processados/Dataset2.csv", sep=','),
            'colunas_discretas': colunas_discretas2,
            }
        },
        'df3':{
            'original': {
            'nome_banco': "DATASET 3 ORIGINAL",
            'data': pd.read_csv(f"Datasets/Processados/Dataset3.csv", sep=','),
            'colunas_discretas': colunas_discretas3,
            }
        },
        'df4':{
            'original': {
            'nome_banco': "DATASET 4 ORIGINAL",
            'data': pd.read_csv(f"Datasets/Processados/Dataset4.csv", sep=','),
            'colunas_discretas': colunas_discretas4,
            }
        }
    }

    # As classes fazem referência a como o smote será aplicado
    variaveis_sensiveis = {
    'sensitive_sexo': {
        'classe_alterada': 1,                   # Aultera o número de mulheres
        'classe_intacta': 0,                    # Não altera o número de homens
        'dados_sensiveis': dados_sensiveis_sexo
    },
    'sensitive_age': {
        'classe_alterada': 1,                   # Altera o número de adultos
        'classe_intacta': 0,                    # Não altera o número de jovens
        'dados_sensiveis': dados_sensiveis_age
    }
    }

    enviesamentos = {
        'smotenc_simples': {
            'funcao': enviesamento_smotenc,
            'nome_banco': 'SMOTE SIMPLES'
        }
    }

    for df in datasets:
        banco = datasets[df]['original']

        # Aplicando pré-processamento
        X_train, X_test, y_train, y_test = aplicar_pre_processador(banco['data'])

        df_treino = pd.concat([X_train, y_train], axis=1)
        df_teste = pd.concat([X_test, y_test], axis=1)

        nome_banco = banco['nome_banco']
        colunas_discretas = banco['colunas_discretas']

        datasets[df].pop('original', None)

        # Criando um dataset exclusivo para cada variável sensível
        for varsen in variaveis_sensiveis:
            colunas_para_remover = [col for col in variaveis_sensiveis if col != varsen]

            novo_nome_banco = nome_banco + " COM " + varsen.upper().replace("_", " ")   # Explicita a variável sensível usada
            novos_dados_treino = df_treino.drop(colunas_para_remover, axis = 1)  # Remove as outras variáveis sensíveis
            novos_dados_teste = df_teste.drop(colunas_para_remover, axis = 1)    # Adiciona a variável sensível a lista de colunas discretas para o smotenc
            novas_colunas_discretas = colunas_discretas + [varsen]
            dados_sensiveis = variaveis_sensiveis[varsen]['dados_sensiveis']

            df_novo = {
            'nome_banco': novo_nome_banco,         
            'treino': novos_dados_treino,
            'teste': novos_dados_teste,
            'colunas_discretas': novas_colunas_discretas, 
            'dados_sensiveis': dados_sensiveis,
            }

            datasets[df][f'original_' + varsen] = df_novo

            # Enviesando os datasets
            for tipo in enviesamentos:
                enviesamento = enviesamentos[tipo]

                funcao = enviesamento['funcao']
                colunas_discretas = banco['colunas_discretas']

                variavel_sensivel = varsen
                classe_alterada = variaveis_sensiveis[varsen]['classe_alterada']
                classe_intacta = variaveis_sensiveis[varsen]['classe_intacta']

                df_enviesado_treino = funcao(novos_dados_treino, colunas_discretas, variavel_sensivel, classe_alterada, classe_intacta, False)
                
                df_enviesado = {
                    'nome_banco': novo_nome_banco.replace("ORIGINAL", enviesamento['nome_banco']),
                    'treino': df_enviesado_treino,
                    'teste': novos_dados_teste,
                    'colunas_discretas': novas_colunas_discretas,
                    'dados_sensiveis': dados_sensiveis
                }

                datasets[df][f'{tipo}_' + varsen] = df_enviesado
    
    return datasets
