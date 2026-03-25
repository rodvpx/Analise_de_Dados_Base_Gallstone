############################################################
# INSTALACAO DE PACOTES (EXECUTAR APENAS UMA VEZ)
############################################################

# install_if_missing <- function(p) {
#   if (!require(p, character.only = TRUE)) {
#     install.packages(p)
#     library(p, character.only = TRUE)
#   }
# }
#
# lapply(c(
#   "dplyr", "ggplot2", "readxl", "caret",
#   "randomForest", "e1071", "xgboost", "pROC",
#   "tidyr", "MASS"
# ), install_if_missing)

############################################################
# 1. PACOTES
############################################################

library(dplyr)
library(readxl)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(pROC)
library(ggplot2)
library(tidyr)
library(MASS)

############################################################
# 2. CARREGAMENTO DA BASE
############################################################

df <- read_excel("dataset-uci.xlsx")
names(df) <- make.names(names(df))

target <- "Gallstone.Status"

############################################################
# 3. PRE-PROCESSAMENTO
############################################################

df[[target]] <- as.factor(df[[target]])
levels(df[[target]]) <- c("No", "Yes")

############################################################
# 4. SPLIT TREINO / TESTE
############################################################

set.seed(123)
train_index <- createDataPartition(df[[target]], p = 0.8, list = FALSE)
train_raw <- df[train_index, ]
test_raw  <- df[-train_index, ]

############################################################
# 5. LIMPEZA DE FEATURES
############################################################

predictor_names <- setdiff(names(train_raw), target)

nzv <- nearZeroVar(train_raw[, predictor_names, drop = FALSE])
if (length(nzv) > 0) {
  predictor_names <- predictor_names[-nzv]
}

num_cols <- sapply(train_raw[, predictor_names, drop = FALSE], is.numeric)
if (sum(num_cols) > 1) {
  cor_mat <- cor(train_raw[, predictor_names[num_cols], drop = FALSE], use = "pairwise.complete.obs")
  high_cor <- findCorrelation(cor_mat, cutoff = 0.9)
  if (length(high_cor) > 0) {
    drop_high_cor <- predictor_names[num_cols][high_cor]
    predictor_names <- setdiff(predictor_names, drop_high_cor)
  }
}

train <- train_raw[, c(target, predictor_names), drop = FALSE]
test  <- test_raw[, c(target, predictor_names), drop = FALSE]

############################################################
# 6. NORMALIZACAO + IMPUTACAO
############################################################

pre_proc <- preProcess(
  train[, predictor_names, drop = FALSE],
  method = c("center", "scale", "knnImpute")
)

train_x <- predict(pre_proc, train[, predictor_names, drop = FALSE])
test_x  <- predict(pre_proc, test[, predictor_names, drop = FALSE])

train <- cbind(train_x, Gallstone.Status = train[[target]])
test  <- cbind(test_x, Gallstone.Status = test[[target]])
names(train)[names(train) == "Gallstone.Status"] <- target
names(test)[names(test) == "Gallstone.Status"] <- target

predictor_names <- setdiff(names(train), target)

############################################################
# 7. CONTROLE DE TREINAMENTO
############################################################

train_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

############################################################
# 8. SELECAO DE VARIAVEIS (BACKWARD)
############################################################

get_step_vars <- function(direction, train_df, target_name) {
  full_formula <- as.formula(paste(target_name, "~ ."))
  null_formula <- as.formula(paste(target_name, "~ 1"))

  full_model <- glm(full_formula, data = train_df, family = binomial())
  null_model <- glm(null_formula, data = train_df, family = binomial())

  step_model <- switch(
    direction,
    forward = stepAIC(null_model,
      scope = list(lower = null_formula, upper = full_formula),
      direction = "forward", trace = FALSE
    ),
    backward = stepAIC(full_model, direction = "backward", trace = FALSE),
    both = stepAIC(full_model, direction = "both", trace = FALSE)
  )

  vars <- setdiff(names(coef(step_model)), "(Intercept)")
  unique(vars)
}

safe_vars <- function(x, fallback) {
  if (is.null(x) || length(x) == 0) return(fallback)
  unique(intersect(fallback, x))
}

all_vars <- predictor_names

vars_backward <- tryCatch(
  get_step_vars("backward", train, target),
  error = function(e) all_vars
)

feature_sets <- list(
  backward = safe_vars(vars_backward, all_vars)
)

feature_sets <- lapply(feature_sets, function(v) {
  if (length(v) < 2) all_vars else v
})

cat("\n=========================================\n")
cat("SELECAO DE VARIAVEIS (PASSO 8)\n")
cat("=========================================\n")
for (nm in names(feature_sets)) {
  cat(nm, "(", length(feature_sets[[nm]]), "): ", paste(feature_sets[[nm]], collapse = ", "), "\n", sep = "")
}
cat("=========================================\n")

############################################################
# 9. XGBOOST OTIMIZADO
############################################################

# Algoritmo XGBoost usado: gbtree (API nativa xgboost)
run_xgb_gbtree <- function(train_df, target_name, vars, seed = 123) {
  set.seed(seed)
  train_sub <- train_df[, c(target_name, vars), drop = FALSE]
  x_train <- as.matrix(train_sub[, vars, drop = FALSE])
  y_train <- ifelse(train_sub[[target_name]] == "Yes", 1, 0)
  dtrain <- xgb.DMatrix(data = x_train, label = y_train)

  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8
  )

  cv_fit <- tryCatch(
    xgb.cv(
      data = dtrain,
      params = params,
      nrounds = 300,
      nfold = 10,
      early_stopping_rounds = 20,
      verbose = 0
    ),
    error = function(e) NULL
  )

  if (is.null(cv_fit)) {
    return(list(model = NULL, auc_cv = -Inf, vars = vars, xgb_method = "gbtree", nrounds = NA_integer_))
  }

  best_nrounds <- cv_fit$best_iteration
  if (is.null(best_nrounds) || length(best_nrounds) == 0 || best_nrounds == 0) {
    auc_col <- if ("test_auc_mean" %in% colnames(cv_fit$evaluation_log)) "test_auc_mean" else "test_auc"
    best_nrounds <- which.max(cv_fit$evaluation_log[[auc_col]])
  }

  model_fit <- tryCatch(
    xgb.train(
      params = params,
      data = dtrain,
      nrounds = best_nrounds,
      verbose = 0
    ),
    error = function(e) NULL
  )

  if (is.null(model_fit)) {
    return(list(model = NULL, auc_cv = -Inf, vars = vars, xgb_method = "gbtree", nrounds = best_nrounds))
  }

  auc_col <- if ("test_auc_mean" %in% colnames(cv_fit$evaluation_log)) "test_auc_mean" else "test_auc"
  eval_df <- as.data.frame(cv_fit$evaluation_log)
  best_nrounds <- max(1, min(best_nrounds, nrow(eval_df)))
  auc_cv <- as.numeric(eval_df[[auc_col]][best_nrounds])

  list(model = model_fit, auc_cv = auc_cv, vars = vars, xgb_method = "gbtree", nrounds = best_nrounds)
}

# testa cada estrategia de selecao e escolhe a melhor para XGBoost gbtree
set.seed(123)
xgb_candidates <- lapply(
  feature_sets,
  function(v) run_xgb_gbtree(train, target, v, seed = 123)
)

candidate_auc <- sapply(xgb_candidates, function(m) m$auc_cv)
if (all(!is.finite(candidate_auc) | candidate_auc == -Inf)) {
  stop("Nao foi possivel treinar XGBoost (gbtree) com as configuracoes atuais.")
}

best_name <- names(which.max(candidate_auc))
best_xgb <- xgb_candidates[[best_name]]

cat("\n=========================================\n")
cat("MELHOR SELECAO DE VARIAVEIS PARA XGBOOST\n")
cat("=========================================\n")
cat("Algoritmo XGBoost usado: xgboost nativo (", best_xgb$xgb_method, ")\n", sep = "")
cat("Melhor estrategia de selecao:", best_name, "\n")
cat("AUC CV:", round(best_xgb$auc_cv, 4), "\n")
cat("Melhor nrounds:", best_xgb$nrounds, "\n")
cat("Numero de variaveis:", length(best_xgb$vars), "\n")
cat("Variaveis escolhidas:\n")
cat(paste(best_xgb$vars, collapse = ", "), "\n")
cat("=========================================\n")

############################################################
# 10. TREINAMENTO RF E SVM (MESMAS VARIAVEIS DO MELHOR XGB)
############################################################

train_best <- train[, c(target, best_xgb$vars), drop = FALSE]
test_best  <- test[, c(target, best_xgb$vars), drop = FALSE]

set.seed(123)
model_rf <- train(
  as.formula(paste(target, "~ .")),
  data = train_best,
  method = "rf",
  trControl = train_control,
  metric = "ROC",
  tuneLength = 10,
  ntree = 500
)

set.seed(123)
model_svm <- train(
  as.formula(paste(target, "~ .")),
  data = train_best,
  method = "svmRadial",
  trControl = train_control,
  metric = "ROC",
  tuneLength = 10
)

############################################################
# 11. FUNCOES DE AVALIACAO
############################################################

avaliar_modelo <- function(model, test_df, target_name, nome, xgb_direct = FALSE, vars = NULL) {
  if (xgb_direct) {
    x_test <- as.matrix(test_df[, vars, drop = FALSE])
    prob <- predict(model, x_test)
    pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels = c("No", "Yes"))
  } else {
    prob <- predict(model, test_df, type = "prob")[, "Yes"]
    pred <- predict(model, test_df)
  }

  cm <- confusionMatrix(pred, test_df[[target_name]])
  roc_obj <- roc(test_df[[target_name]], prob)

  cat("\n====================\n", nome, "\n====================\n")
  print(cm)
  cat("AUC:", auc(roc_obj), "\n")

  list(cm = cm, auc = auc(roc_obj), prob = prob, pred = pred)
}

f1_score <- function(cm) {
  precision <- as.numeric(cm$byClass["Pos Pred Value"])
  recall <- as.numeric(cm$byClass["Sensitivity"])
  if ((precision + recall) == 0) return(0)
  2 * (precision * recall) / (precision + recall)
}

############################################################
# 12. AVALIACAO FINAL
############################################################

res_rf <- avaliar_modelo(model_rf, test_best, target, "Random Forest")
res_svm <- avaliar_modelo(model_svm, test_best, target, "SVM")
res_xgb <- avaliar_modelo(
  model = best_xgb$model,
  test_df = test_best,
  target_name = target,
  nome = paste0("XGBoost-gbtree (", best_name, ")"),
  xgb_direct = TRUE,
  vars = best_xgb$vars
)

############################################################
# 13. TABELA FINAL DE COMPARACAO (SEM STACKING)
############################################################

comparacao <- data.frame(
  Modelo = c("Random Forest", "SVM", paste0("XGBoost-", best_name)),
  Accuracy = c(
    as.numeric(res_rf$cm$overall["Accuracy"]),
    as.numeric(res_svm$cm$overall["Accuracy"]),
    as.numeric(res_xgb$cm$overall["Accuracy"])
  ),
  AUC = c(
    as.numeric(res_rf$auc),
    as.numeric(res_svm$auc),
    as.numeric(res_xgb$auc)
  ),
  Sensitivity = c(
    as.numeric(res_rf$cm$byClass["Sensitivity"]),
    as.numeric(res_svm$cm$byClass["Sensitivity"]),
    as.numeric(res_xgb$cm$byClass["Sensitivity"])
  ),
  Specificity = c(
    as.numeric(res_rf$cm$byClass["Specificity"]),
    as.numeric(res_svm$cm$byClass["Specificity"]),
    as.numeric(res_xgb$cm$byClass["Specificity"])
  ),
  Precision = c(
    as.numeric(res_rf$cm$byClass["Pos Pred Value"]),
    as.numeric(res_svm$cm$byClass["Pos Pred Value"]),
    as.numeric(res_xgb$cm$byClass["Pos Pred Value"])
  ),
  F1 = c(
    f1_score(res_rf$cm),
    f1_score(res_svm$cm),
    f1_score(res_xgb$cm)
  )
)

comparacao <- comparacao %>%
  mutate(across(-Modelo, \(x) round(x, 3)))

print(comparacao)

############################################################
# 14. CURVAS ROC COMPARATIVAS (SEM STACKING)
############################################################

roc_rf <- roc(test_best[[target]], res_rf$prob)
roc_svm <- roc(test_best[[target]], res_svm$prob)
roc_xgb <- roc(test_best[[target]], res_xgb$prob)

plot(roc_rf, col = "blue", lwd = 2, main = "Curvas ROC - Comparacao")
lines(roc_svm, col = "green", lwd = 2)
lines(roc_xgb, col = "red", lwd = 2)

legend("bottomright",
  legend = c(
    paste0("RF (AUC = ", round(auc(roc_rf), 3), ")"),
    paste0("SVM (AUC = ", round(auc(roc_svm), 3), ")"),
    paste0("XGB-", best_name, " (AUC = ", round(auc(roc_xgb), 3), ")")
  ),
  col = c("blue", "green", "red"),
  lwd = 2,
  cex = 0.9
)

############################################################
# 15. COMPARACAO VISUAL DAS METRICAS
############################################################

comparacao_long <- comparacao %>%
  pivot_longer(
    cols = -Modelo,
    names_to = "Metrica",
    values_to = "Valor"
  )

ggplot(comparacao_long, aes(x = Modelo, y = Valor, fill = Metrica)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(
    title = "Comparacao de Modelos",
    y = "Valor",
    x = "Modelo"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

############################################################
# 16. COMPARACAO DE AUC
############################################################

ggplot(comparacao, aes(x = Modelo, y = AUC, fill = Modelo)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Comparacao de AUC entre Modelos") +
  ylim(0, 1)

############################################################
# 17. IMPORTANCIA DAS VARIAVEIS ESCOLHIDAS (XGBOOST)
############################################################

imp_xgb <- xgb.importance(
  model = best_xgb$model,
  feature_names = best_xgb$vars
)

if (is.null(imp_xgb) || nrow(imp_xgb) == 0) {
  cat("\nNao foi possivel gerar importancia de variaveis do XGBoost.\n")
} else {
  imp_plot <- as.data.frame(imp_xgb) %>%
    mutate(Feature = factor(Feature, levels = rev(Feature)))

  ggplot(imp_plot, aes(x = Feature, y = Gain)) +
    geom_col(fill = "#1f77b4") +
    coord_flip() +
    theme_minimal() +
    labs(
      title = "Importancia das Variaveis Selecionadas (XGBoost gbtree)",
      x = "Variavel",
      y = "Gain"
    )
}
