# Predição de Doença da Vesícula Biliar

Este repositório contém o pipeline de machine learning em R para classificação de `Gallstone.Status`, com comparação entre modelos base e versões com threshold otimizado.

## Métricas finais (modelos base)

| Modelo | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.873 | 0.951 | 0.906 | 0.839 | 0.853 | 0.879 |
| XGBoost | 0.857 | 0.927 | 0.906 | 0.806 | 0.829 | 0.866 |
| SVM | 0.730 | 0.802 | 0.812 | 0.645 | 0.703 | 0.754 |

## Melhorias (threshold otimizado)

| Modelo | Threshold | Accuracy | Sensitivity | Specificity | AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Blend RF+XGB (threshold otimizado) | 0.439 | 0.889 | 0.906 | 0.871 | 0.938 | 0.892 |
| RF (threshold otimizado) | 0.477 | 0.873 | 0.906 | 0.839 | 0.951 | 0.879 |
| XGBoost (threshold otimizado) | 0.405 | 0.873 | 0.906 | 0.839 | 0.927 | 0.879 |

## Gráficos

### Comparação de algoritmos
![Comparação dos algoritmos](img/Comparacao%20dos%20algoritimos.png)

### ROC do melhor modelo
![ROC melhor modelo](img/Roc%20Melhor%20modelo.png)

### Matriz de confusão do melhor modelo
![Matriz de confusão](img/Matriz%20de%20confusao.png)

### Distribuição de probabilidades
![Distribuição de probabilidades](img/Distribuicao%20de%20probabilidades.png)

### Importância das variáveis
![Importância das variáveis](img/importacia%20das%20variaveis.png)

