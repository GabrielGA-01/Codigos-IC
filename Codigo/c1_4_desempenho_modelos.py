import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

def desempenho_modelo(nome_base_de_dados, model, X_test, X_test_justica, y_pred, y_test, dados_sensiveis, printar=False, matriz_de_confusao=False):
    coluna_sensivel = dados_sensiveis['coluna_sensivel']
    grupo_privilegiado = dados_sensiveis['grupo_privilegiado']
    grupo_desprivilegiado = dados_sensiveis['grupo_desprivilegiado']

    # Para os resultados
    desempenho_geral = {}
    desempenho_privilegiado = {}
    desempenho_desprivilegiado = {}

    # Separando os grupos
    filtro_privilegiado = X_test_justica[coluna_sensivel] == grupo_privilegiado
    y_test_priv = y_test[filtro_privilegiado]
    y_pred_priv = y_pred[filtro_privilegiado]
    X_test_priv = X_test[filtro_privilegiado]

    filtro_desprivilegiado = X_test_justica[coluna_sensivel] == grupo_desprivilegiado
    y_test_despriv = y_test[filtro_desprivilegiado]
    y_pred_despriv = y_pred[filtro_desprivilegiado]
    X_test_despriv = X_test[filtro_desprivilegiado]

    # RELATÓRIO DE CLASSIFICAÇÃO
    desempenho_geral['relatorio_classificacao'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    desempenho_privilegiado['relatorio_classificacao'] = classification_report(y_test_priv, y_pred_priv, output_dict=True, zero_division=0)
    desempenho_desprivilegiado['relatorio_classificacao'] = classification_report(y_test_despriv, y_pred_despriv, output_dict=True, zero_division=0)

    if printar:
        print(f'\n----- {nome_base_de_dados} || RELATÓRIO DE CLASSIFICAÇÃO -----\n')
        print(classification_report(y_test, y_pred, zero_division=0))

    # CÁLCULO DA AUC
    y_pred_proba_geral = model.predict_proba(X_test)[:, 1]
    y_pred_proba_priv = model.predict_proba(X_test_priv)[:, 1]
    y_pred_proba_despriv = model.predict_proba(X_test_despriv)[:, 1]

    try:
        desempenho_geral['ROC_AUC'] = roc_auc_score(y_test, y_pred_proba_geral)
    except ValueError:
        desempenho_geral['ROC_AUC'] = 0.0
    
    try:
        desempenho_privilegiado['ROC_AUC'] = roc_auc_score(y_test_priv, y_pred_proba_priv)
    except ValueError:
         desempenho_privilegiado['ROC_AUC'] = 0.0
    
    try:
        desempenho_desprivilegiado['ROC_AUC'] = roc_auc_score(y_test_despriv, y_pred_proba_despriv)
    except ValueError:
        desempenho_desprivilegiado['ROC_AUC'] = 0.0

    if printar:
        print(f"ROC AUC: {desempenho_geral['ROC_AUC']:.4f}")

    # CÁLCULO DO F1 SCORE
    desempenho_geral['F1_Score'] = f1_score(y_test, y_pred, zero_division=0)
    desempenho_privilegiado['F1_Score'] = f1_score(y_test_priv, y_pred_priv, zero_division=0)
    desempenho_desprivilegiado['F1_Score'] = f1_score(y_test_despriv, y_pred_despriv, zero_division=0)

    if printar:
        print(f"F1 Score: {desempenho_geral['F1_Score']:.4f}")

    # Função interna para evitar que o código falhe se um grupo não tiver as duas classes
    def calcular_ks_seguro(y_true, y_proba):
        proba_1 = y_proba[y_true == 1]
        proba_0 = y_proba[y_true == 0]
        
        # Só calcula se existirem exemplos de ambas as classes
        if len(proba_1) > 0 and len(proba_0) > 0:
            return ks_2samp(proba_1, proba_0).statistic
        return 0.0

    desempenho_geral['KS'] = calcular_ks_seguro(y_test, y_pred_proba_geral)
    desempenho_privilegiado['KS'] = calcular_ks_seguro(y_test_priv, y_pred_proba_priv)
    desempenho_desprivilegiado['KS'] = calcular_ks_seguro(y_test_despriv, y_pred_proba_despriv)

    if printar:
        print(f"KS: {desempenho_geral['KS']:.4f}")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    desempenho_geral['matriz_de_confusao'] = cm.tolist()
    desempenho_privilegiado['matriz_de_confusao'] = confusion_matrix(y_test_priv, y_pred_priv).tolist()
    desempenho_desprivilegiado['matriz_de_confusao'] = confusion_matrix(y_test_despriv, y_pred_despriv).tolist()

    if matriz_de_confusao:
        print(f'\n----- {nome_base_de_dados} || MATRIZ DE CONFUSÃO -----\n')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['0', '1'],
                    yticklabels=['0', '1'])
        plt.title('Matriz de Confusão')
        plt.show()
    elif printar:
        print(f'\n----- {nome_base_de_dados} || MATRIZ DE CONFUSÃO -----\n')
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f'TN: {tn} | FP: {fp}')
            print(f'FN: {fn} | TP: {tp}')

    # Verificando a justiça do modelo
    if printar:
        print(f'\n----- {nome_base_de_dados} || JUSTIÇA DO MODELO -----\n')

    df_results = pd.DataFrame({
        'Y_real': y_test,
        'Y_predito': y_pred,
        coluna_sensivel: X_test_justica[coluna_sensivel]
    })

    # Separando os grupos
    group_desprivilegiado = (df_results[coluna_sensivel] == grupo_desprivilegiado)
    group_privilegiado = (df_results[coluna_sensivel] == grupo_privilegiado)

    # Métrica 1 - Demographic Parity
    selection_rate_desprivilegiado = df_results[group_desprivilegiado]['Y_predito'].mean()
    selection_rate_privilegiado = df_results[group_privilegiado]['Y_predito'].mean()

    if pd.isna(selection_rate_desprivilegiado): selection_rate_desprivilegiado = 0.0
    if pd.isna(selection_rate_privilegiado): selection_rate_privilegiado = 0.0

    demographic_parity_difference = selection_rate_desprivilegiado - selection_rate_privilegiado
    demographic_parity_ratio = (selection_rate_desprivilegiado / selection_rate_privilegiado) if selection_rate_privilegiado > 0 else float('inf')

    if printar:
        print('----- Demographic Parity (Selection Rate) -----')
        print(f'Selection Rate (Desprivilegiado): {selection_rate_desprivilegiado:.4f}')
        print(f'Selection Rate (Privilegiado):   {selection_rate_privilegiado:.4f}')
        print(f'Demographic Parity Difference: {demographic_parity_difference:.4f}')
        print(f'Demographic Parity Ratio:    {demographic_parity_ratio:.4f}\n')

    desempenho_geral['demographic_parity'] = {
        'selection_rate_privilegiado': selection_rate_privilegiado,
        'selection_rate_desprivilegiado': selection_rate_desprivilegiado,
        'demographic_parity_difference': demographic_parity_difference,
        'demographic_parity_ratio': demographic_parity_ratio
    }

    # Metrica 2 - True Positive Rate (Equal Opportunity)
    positives_real = df_results[df_results['Y_real'] == 1]
    tpr_desprivilegiado = positives_real[positives_real[coluna_sensivel] == grupo_desprivilegiado]['Y_predito'].mean()
    tpr_privilegiado = positives_real[positives_real[coluna_sensivel] == grupo_privilegiado]['Y_predito'].mean()

    if pd.isna(tpr_desprivilegiado): tpr_desprivilegiado = 0.0
    if pd.isna(tpr_privilegiado): tpr_privilegiado = 0.0

    true_positive_rate_difference = tpr_desprivilegiado - tpr_privilegiado
    true_positive_rate_ratio = (tpr_desprivilegiado / tpr_privilegiado) if tpr_privilegiado > 0 else float('inf')

    if printar:
        print('----- True Positive Rate (Equal Opportunity) -----')
        print(f'TPR (Desprivilegiado): {tpr_desprivilegiado:.4f}')
        print(f'TPR (Privilegiado):   {tpr_privilegiado:.4f}')
        print(f'Diff: {true_positive_rate_difference:.4f} | Ratio: {true_positive_rate_ratio:.4f}\n')

    desempenho_geral['true_positive_rate'] = {
        'true_positive_rate_privilegiado': tpr_privilegiado,
        'true_positive_rate_desprivilegiado': tpr_desprivilegiado,
        'true_positive_rate_difference': true_positive_rate_difference,
        'true_positive_rate_ratio': true_positive_rate_ratio
    }

    # Metrica 3 - False Positive Rate
    negatives_real = df_results[df_results['Y_real'] == 0]
    fpr_desprivilegiado = negatives_real[negatives_real[coluna_sensivel] == grupo_desprivilegiado]['Y_predito'].mean()
    fpr_privilegiado = negatives_real[negatives_real[coluna_sensivel] == grupo_privilegiado]['Y_predito'].mean()

    if pd.isna(fpr_desprivilegiado): fpr_desprivilegiado = 0.0
    if pd.isna(fpr_privilegiado): fpr_privilegiado = 0.0

    fpr_difference = fpr_desprivilegiado - fpr_privilegiado
    fpr_ratio = (fpr_desprivilegiado / fpr_privilegiado) if fpr_privilegiado > 0 else float('inf')

    if printar:
        print('----- False Positive Rate -----')
        print(f'FPR (Desprivilegiado): {fpr_desprivilegiado:.4f}')
        print(f'FPR (Privilegiado):   {fpr_privilegiado:.4f}')
        print(f'Diff: {fpr_difference:.4f} | Ratio: {fpr_ratio:.4f}\n')

    desempenho_geral['false_positive_rate'] = {
        'false_positive_rate_privilegiado': fpr_privilegiado,
        'false_positive_rate_desprivilegiado': fpr_desprivilegiado,
        'false_positive_rate_difference': fpr_difference,
        'false_positive_rate_ratio': fpr_ratio
    }

    # Metrica 4 - False Negative Rate
    fnr_desprivilegiado = 1 - tpr_desprivilegiado
    fnr_privilegiado = 1 - tpr_privilegiado

    fnr_difference = fnr_desprivilegiado - fnr_privilegiado
    fnr_ratio = (fnr_desprivilegiado / fnr_privilegiado) if fnr_privilegiado > 0 else float('inf')

    if printar:
        print('----- False Negative Rate -----')
        print(f'FNR Diff: {fnr_difference:.4f} | Ratio: {fnr_ratio:.4f}\n')

    desempenho_geral['false_negative_rate'] = {
        'false_negative_rate_privilegiado': fnr_privilegiado,
        'false_negative_rate_desprivilegiado': fnr_desprivilegiado,
        'false_negative_rate_difference': fnr_difference,
        'false_negative_rate_ratio': fnr_ratio
    }

    # Metrica 5 - Predictive Parity (Precision)
    positives_predicted = df_results[df_results['Y_predito'] == 1]
    precision_desprivilegiado = positives_predicted[positives_predicted[coluna_sensivel] == grupo_desprivilegiado]['Y_real'].mean()
    precision_privilegiado = positives_predicted[positives_predicted[coluna_sensivel] == grupo_privilegiado]['Y_real'].mean()

    # Caso não haja nenhuma previsão positiva
    if pd.isna(precision_desprivilegiado): precision_desprivilegiado = 0.0
    if pd.isna(precision_privilegiado): precision_privilegiado = 0.0

    predictive_parity_difference = precision_desprivilegiado - precision_privilegiado
    predictive_parity_ratio = (precision_desprivilegiado / precision_privilegiado) if precision_privilegiado > 0 else float('inf')

    if printar:
        print('----- Predictive Parity (Precision) -----')
        print(f'Precision (Desprivilegiado): {precision_desprivilegiado:.4f}')
        print(f'Precision (Privilegiado):   {precision_privilegiado:.4f}')
        print(f'Diff: {predictive_parity_difference:.4f} | Ratio: {predictive_parity_ratio:.4f}\n')

    desempenho_geral['predictive_parity'] = {
        'precision_privilegiado': precision_privilegiado,
        'precision_desprivilegiado': precision_desprivilegiado,
        'predictive_parity_difference': predictive_parity_difference,
        'predictive_parity_ratio': predictive_parity_ratio
    }

    # Metrica 6 - Equalized Odds
    equalized_odds_difference = max(abs(tpr_desprivilegiado - tpr_privilegiado), abs(fpr_desprivilegiado - fpr_privilegiado))

    if tpr_privilegiado > 0 and fpr_privilegiado > 0:
        tpr_r = tpr_desprivilegiado / tpr_privilegiado
        fpr_r = fpr_desprivilegiado / fpr_privilegiado
        equalized_odds_ratio = min(tpr_r, fpr_r)
    else:
        equalized_odds_ratio = 0.0

    if printar:
        print('----- Equalized Odds -----')
        print(f'Diff: {equalized_odds_difference:.4f}')
        print(f'Ratio: {equalized_odds_ratio:.4f}\n')

    desempenho_geral['equalized_odds'] = {
        'equalized_odds_difference': equalized_odds_difference,
        'equalized_odds_ratio': equalized_odds_ratio,
    }

    desempenho = {}
    desempenho['geral'] = desempenho_geral
    desempenho['privilegiado'] = desempenho_privilegiado
    desempenho['desprivilegiado'] = desempenho_desprivilegiado

    return desempenho