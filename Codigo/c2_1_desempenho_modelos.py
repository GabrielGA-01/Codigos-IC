# Função auxiliar para calcular as taxas de verdadeiro/falso positivo/negativo

def metricas_matriz_de_confusao(cm):
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    selection_rate = (tp + fp) / total if total > 0 else 0

    return {
        "matriz_de_confusao": cm.tolist(),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "selection_rate": selection_rate
    }

# Função responsável por calcular o desempenho de todos os modelos
# nome_base_de_dados - Nome do conjunto de dados usado nos prints
# model - Modelo já treinado previamente
# X_test - As features do conjunto de teste
# X_test_justica - A coluna com os dados sensíveis
# y_pred - As previsões feitas pelo modelo no conjunto de teste
# y_test - Os valores reais dos rótulos do conjunto de teste
# dados_sensiveis - Um dicionário contendo o nome da coluna sensível e o grupo privilegiado e desprivilegiado
# matriz_de_confusao - Determina se será feito e exibido um gráfico com a matriz de confusão

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

  desempenho = {}

  # Relatório de classificação
  if(printar):
    print(f"\n----- {nome_base_de_dados} || RELATÓRIO DE CLASSIFICAÇÃO -----\n")
    print(classification_report(y_test, y_pred, zero_division=0))
  desempenho["relatorio_classificacao"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

  # Cálculo da AUC
  y_pred_proba = model.predict_proba(X_test)[:, 1]
  auc = roc_auc_score(y_test, y_pred_proba)
  desempenho['ROC_AUC'] = auc
  if(printar):
    print(f"ROC AUC: {auc:.4f}")

  # Cálculo do F1 Score
  f1 = f1_score(y_test, y_pred, zero_division=0)
  desempenho['F1_Score'] = f1
  if(printar):
    print(f"F1 Score: {f1:.4f}")

  # Cálculo do KS
  proba_1 = y_pred_proba[y_test == 1]
  proba_0 = y_pred_proba[y_test == 0]
  ks = ks_2samp(proba_1, proba_0).statistic
  desempenho['KS'] = ks
  if(printar):
    print(f"KS: {ks:.4f}")

  # Matriz de Confusão
  cm = confusion_matrix(y_test, y_pred)
  desempenho["matriz_de_confusao"] = cm.tolist()
  if(matriz_de_confusao):
    print(f"\n----- {nome_base_de_dados} || MATRIZ DE CONFUSÃO -----\n")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'],
                yticklabels=['0', '1'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Rótulo Verdadeiro (True Label)')
    plt.xlabel('Rótulo Previsto (Predicted Label)')
    plt.show()
  elif(printar):
    print(f"\n----- {nome_base_de_dados} || MATRIZ DE CONFUSÃO -----\n")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"TN: {tn} | FP: {fp}")
        print(f"FN: {fn} | TP: {tp}")

  # Para a variavel sensivel
  filtro_privilegiado = X_test_justica[coluna_sensivel] == grupo_privilegiado
  y_test_priv = y_test[filtro_privilegiado]
  y_pred_priv = y_pred[filtro_privilegiado]
  cm_priv = confusion_matrix(y_test_priv, y_pred_priv)
  desempenho[f'{coluna_sensivel}_privilegiado'] = metricas_matriz_de_confusao(cm_priv)

  filtro_desprivilegiado = X_test_justica[coluna_sensivel] == grupo_desprivilegiado
  y_test_despriv = y_test[filtro_desprivilegiado]
  y_pred_despriv = y_pred[filtro_desprivilegiado]
  cm_despriv = confusion_matrix(y_test_despriv, y_pred_despriv)
  desempenho[f'{coluna_sensivel}_desprivilegiado'] = metricas_matriz_de_confusao(cm_despriv)

  # Verificando a justiça do modelo
  if(printar):
    print(f"\n----- {nome_base_de_dados} || JUSTIÇA DO MODELO -----\n")

  df_results = pd.DataFrame({
    'Y_real': y_test,
    'Y_predito': y_pred,
    coluna_sensivel: X_test_justica[coluna_sensivel]
  })

  # Separando os grupos
  group_desprivilegiado = (df_results[coluna_sensivel] == grupo_desprivilegiado)
  group_privilegiado = (df_results[coluna_sensivel] == grupo_privilegiado)

  # Métrica 1 - Demographic Parity (Selection Rate) - (Paridade estatística)
  selection_rate_desprivilegiado = df_results[group_desprivilegiado]['Y_predito'].mean()
  selection_rate_privilegiado = df_results[group_privilegiado]['Y_predito'].mean()

  demographic_parity_difference = selection_rate_desprivilegiado - selection_rate_privilegiado

  if selection_rate_privilegiado > 0:
      demographic_parity_ratio = selection_rate_desprivilegiado / selection_rate_privilegiado
  else:
      demographic_parity_ratio = float('inf')

  if(printar):
      print("----- Demographic Parity (Selection Rate) -----")
      print(f"Selection Rate (Desprivilegiado): {selection_rate_desprivilegiado:.4f}")
      print(f"Selection Rate (Privilegiado):   {selection_rate_privilegiado:.4f}")
      print(f"Demographic Parity Difference: {demographic_parity_difference:.4f}")
      print(f"Demographic Parity Ratio:    {demographic_parity_ratio:.4f}\n")

  desempenho["demographic_parity"] = {
      "selection_rate_privilegiado": selection_rate_privilegiado,
      "selection_rate_desprivilegiado": selection_rate_desprivilegiado,
      "demographic_parity_difference": demographic_parity_difference,
      "demographic_parity_ratio": demographic_parity_ratio
  }

  # Metrica 2 - True Positive Rate - (Equal Opportunity)
  positives_real = df_results[df_results['Y_real'] == 1]
  tpr_desprivilegiado = positives_real[positives_real[coluna_sensivel] == grupo_desprivilegiado]['Y_predito'].mean()
  tpr_privilegiado = positives_real[positives_real[coluna_sensivel] == grupo_privilegiado]['Y_predito'].mean()

  true_positive_rate_difference = tpr_desprivilegiado - tpr_privilegiado

  if tpr_privilegiado > 0:
      true_positive_rate_ratio = tpr_desprivilegiado / tpr_privilegiado
  else:
      true_positive_rate_ratio = float('inf')

  if(printar):
      print("----- True Positive Rate (Equal Opportunity) -----")
      print(f"True Positive Rate (Desprivilegiado): {tpr_desprivilegiado:.4f}")
      print(f"True Positive Rate (Privilegiado):   {tpr_privilegiado:.4f}")
      print(f"True Positive Rate Difference: {true_positive_rate_difference:.4f}")
      print(f"True Positive Rate Ratio:      {true_positive_rate_ratio:.4f}\n")

  desempenho["true_positive_rate"] = {
      "true_positive_rate_privilegiado": tpr_privilegiado,
      "true_positive_rate_desprivilegiado": tpr_desprivilegiado,
      "true_positive_rate_difference": true_positive_rate_difference,
      "true_positive_rate_ratio": true_positive_rate_ratio
  }

  # Metrica 3 - False Positive Rate
  negatives_real = df_results[df_results['Y_real'] == 0]
  fpr_desprivilegiado = negatives_real[negatives_real[coluna_sensivel] == grupo_desprivilegiado]['Y_predito'].mean()
  fpr_privilegiado = negatives_real[negatives_real[coluna_sensivel] == grupo_privilegiado]['Y_predito'].mean()

  fpr_difference = fpr_desprivilegiado - fpr_privilegiado

  if fpr_privilegiado > 0:
      fpr_ratio = fpr_desprivilegiado / fpr_privilegiado
  else:
      fpr_ratio = float('inf')

  if(printar):
      print("----- False Positive Rate -----")
      print(f"False Positive Rate (Desprivilegiado): {fpr_desprivilegiado:.4f}")
      print(f"False Positive Rate (Privilegiado):   {fpr_privilegiado:.4f}")
      print(f"False Positive Rate Difference: {fpr_difference:.4f}")
      print(f"False Positive Rate Ratio:      {fpr_ratio:.4f}\n")

  desempenho["false_positive_rate"] = {
      "false_positive_rate_privilegiado": fpr_privilegiado,
      "false_positive_rate_desprivilegiado": fpr_desprivilegiado,
      "false_positive_rate_difference": fpr_difference,
      "false_positive_rate_ratio": fpr_ratio
  }

  # Metrica 4 - False Negative Rate
  fnr_desprivilegiado = 1 - tpr_desprivilegiado
  fnr_privilegiado = 1 - tpr_privilegiado

  fnr_difference = fnr_desprivilegiado - fnr_privilegiado

  if fnr_privilegiado > 0:
      fnr_ratio = fnr_desprivilegiado / fnr_privilegiado
  else:
      fnr_ratio = float('inf')

  if(printar):
      print("----- False Negative Rate -----")
      print(f"False Negative Rate (Desprivilegiado): {fnr_desprivilegiado:.4f}")
      print(f"False Negative Rate (Privilegiado):   {fnr_privilegiado:.4f}")
      print(f"False Negative Rate Difference: {fnr_difference:.4f}")
      print(f"False Negative Rate Ratio:      {fnr_ratio:.4f}\n")

  desempenho["false_negative_rate"] = {
      "false_negative_rate_privilegiado": fnr_privilegiado,
      "false_negative_rate_desprivilegiado": fnr_desprivilegiado,
      "false_negative_rate_difference": fnr_difference,
      "false_negative_rate_ratio": fnr_ratio
  }

  # Metrica 5 - Predictive Parity (Precision)
  positives_predicted = df_results[df_results['Y_predito'] == 1]
  precision_desprivilegiado = positives_predicted[positives_predicted[coluna_sensivel] == grupo_desprivilegiado]['Y_real'].mean()
  precision_privilegiado = positives_predicted[positives_predicted[coluna_sensivel] == grupo_privilegiado]['Y_real'].mean()

  predictive_parity_difference = precision_desprivilegiado - precision_privilegiado

  if precision_privilegiado > 0:
      predictive_parity_ratio = precision_desprivilegiado / precision_privilegiado
  else:
      predictive_parity_ratio = float('inf')

  if(printar):
      print("----- Predictive Parity (Precision) -----")
      print(f"Precision (Desprivilegiado): {precision_desprivilegiado:.4f}")
      print(f"Precision (Privilegiado):   {precision_privilegiado:.4f}")
      print(f"Predictive Parity Difference: {predictive_parity_difference:.4f}")
      print(f"Predictive Parity Ratio:      {predictive_parity_ratio:.4f}\n")

  desempenho["predictive_parity"] = {
      "precision_privilegiado": precision_privilegiado,
      "precision_desprivilegiado": precision_desprivilegiado,
      "predictive_parity_difference": predictive_parity_difference,
      "predictive_parity_ratio": predictive_parity_ratio
  }

  # Metrica 6 - Equalized Odds
  equalized_odds_difference = max(abs(tpr_desprivilegiado - tpr_privilegiado), abs(fpr_desprivilegiado - fpr_privilegiado))

  if tpr_privilegiado > 0 and fpr_privilegiado > 0:
      tpr_r = tpr_desprivilegiado / tpr_privilegiado
      fpr_r = fpr_desprivilegiado / fpr_privilegiado
      equalized_odds_ratio = min(tpr_r, fpr_r)
  else:
      equalized_odds_ratio = 0

  if(printar):
      print("----- Equalized Odds -----")
      print(f"Equalized Odds Difference (max of TPR/FPR diffs): {equalized_odds_difference:.4f}")
      print(f"Equalized Odds Ratio (min of TPR/FPR ratios):    {equalized_odds_ratio:.4f}\n")

  desempenho["equalized_odds"] = {
      "equalized_odds_difference": equalized_odds_difference,
      "equalized_odds_ratio": equalized_odds_ratio,
  }

  return(desempenho)