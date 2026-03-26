# Predição de Doença da Vesícula Biliar

Pipeline de Machine Learning em R para classificação de `Gallstone.Status`, com reprodução e aprimoramento metodológico de um estudo anterior conduzido por colegas acadêmicos, utilizando a base Gallstone Disease Prediction (UCI).

---

## 🎯 Objetivo

Este projeto tem como objetivo **reproduzir e aprimorar metodologicamente** um estudo anterior de predição de colelitíase utilizando a mesma base de dados, buscando superar os resultados obtidos originalmente por meio de melhorias no pipeline de Machine Learning.

As principais melhorias propostas foram:
- Pipeline de pré-processamento mais robusto
- Seleção automática de variáveis (StepAIC)
- Validação cruzada repetida otimizada por AUC
- Otimização do limiar de classificação (Youden Index)
- Ensemble por *blending* (Random Forest + XGBoost)

---

## 🏆 Principais Resultados

O modelo **Blend RF + XGBoost com threshold otimizado** apresentou o melhor desempenho:

- **Accuracy:** 88.9%
- **AUC:** 0.938
- **F1-Score:** 0.892
- **Sensibilidade:** 90.6%
- **Especificidade:** 87.1%

O estudo superou o trabalho de referência em **+4,78 pontos percentuais de acurácia**.

---

## ⚙️ Pipeline de Machine Learning

O fluxo completo do projeto segue as etapas abaixo:

1. **Particionamento estratificado**
   - Split 80/20 preservando proporção das classes

2. **Pré-processamento**
   - Remoção de variáveis near zero variance
   - Remoção de multicolinearidade (|cor| > 0.90)
   - Normalização (center + scale)
   - Imputação de valores faltantes via KNN

3. **Seleção de variáveis**
   - Stepwise backward (StepAIC)

4. **Treinamento de modelos**
   - Random Forest
   - XGBoost
   - SVM Radial

5. **Validação**
   - Repeated Cross-Validation (10 folds × 3)
   - Métrica principal: AUC

6. **Otimizações**
   - Ajuste do limiar via índice de Youden
   - Ensemble RF + XGBoost (blending)

---

## Dataset

Este projeto utiliza a base **Gallstone Disease Prediction**, disponível no UCI Machine Learning Repository.

🔗 https://archive.ics.uci.edu/dataset/1150/gallstone-1

### Descrição

A base contém dados clínicos e laboratoriais de pacientes utilizados para a **predição de colelitíase (cálculos biliares)** por meio de técnicas de Machine Learning.

O objetivo é classificar se o paciente possui ou não a doença com base em variáveis clínicas, metabólicas e de composição corporal.

### Tipo de problema
- **Tarefa:** Classificação binária  
- **Variável alvo:** `Gallstone.Status` (Yes / No)

### Dimensão da base
- **Número de instâncias:** 319 pacientes  
- **Número de variáveis:** 38 atributos + 1 variável alvo  
- **Tipo de atributos:** Mistos (numéricos e categóricos)

### Exemplos de variáveis
- Proteína C-Reativa (CRP)
- Vitamina D
- Diabetes
- Hiperlipidemia
- Massa óssea
- Gordura corporal total
- Doença arterial coronariana

### Desafios da base
- Presença de valores faltantes
- Variáveis categóricas e numéricas misturadas
- Possível multicolinearidade entre biomarcadores
- Base de tamanho moderado (small tabular dataset)

---

## Métricas finais (modelos base)

| Modelo | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.873 | 0.951 | 0.906 | 0.839 | 0.853 | 0.879 |
| XGBoost | 0.857 | 0.927 | 0.906 | 0.806 | 0.829 | 0.866 |
| SVM | 0.730 | 0.802 | 0.812 | 0.645 | 0.703 | 0.754 |

---

## Melhorias (threshold otimizado)

| Modelo | Threshold | Accuracy | Sensitivity | Specificity | AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Blend RF+XGB (threshold otimizado) | 0.439 | 0.889 | 0.906 | 0.871 | 0.938 | 0.892 |
| RF (threshold otimizado) | 0.477 | 0.873 | 0.906 | 0.839 | 0.951 | 0.879 |
| XGBoost (threshold otimizado) | 0.405 | 0.873 | 0.906 | 0.839 | 0.927 | 0.879 |

---

## Gráficos

### Importância das variáveis
![Importância das variáveis](<img/Grafico - Importancia das Variaveis Selecionadas.png>)

### Comparação de algoritmos (base)
![Comparação dos algoritmos base](<img/Grafico - Comparacao Algoritmos Base.png>)

### Comparação de algoritmos (threshold otimizado)
![Comparação dos algoritmos com threshold](<img/Grafico - Comparacao Algoritmos com Threshold.png>)

### ROC do melhor modelo
![ROC melhor modelo](<img/Grafico - ROC Melhor Modelo.png>)

### Matriz de confusão do melhor modelo
![Matriz de confusão](<img/Grafico - Matriz de Confusao.png>)

### Distribuição de probabilidades
![Distribuição de probabilidades](<img/Grafico - Distribuicao De Probabilidades.png>)

### Impacto do limiar de classificação
![Impacto do limiar de classificação](<img/Grafico - Impacto Do Limiar De Classificação.png>)
