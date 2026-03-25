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
# 8. SELECAO DE VARIAVEIS (FORWARD/BACKWARD/STEPWISE/RFE/GA)
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

vars_forward <- tryCatch(
  get_step_vars("forward", train, target),
  error = function(e) all_vars
)

vars_backward <- tryCatch(
  get_step_vars("backward", train, target),
  error = function(e) all_vars
)

vars_stepwise <- tryCatch(
  get_step_vars("both", train, target),
  error = function(e) all_vars
)

set.seed(123)
rfe_sizes <- unique(sort(pmin(c(3, 5, 7, 10, 15, 20), length(all_vars))))
rfe_sizes <- rfe_sizes[rfe_sizes >= 2]

vars_rfe <- tryCatch({
  rfe_ctrl <- rfeControl(functions = caretFuncs, method = "repeatedcv", number = 5, repeats = 2)
  rfe_fit <- rfe(
    x = train[, all_vars, drop = FALSE],
    y = train[[target]],
    sizes = rfe_sizes,
    rfeControl = rfe_ctrl,
    method = "rf",
    metric = "ROC",
    trControl = trainControl(
      method = "cv",
      number = 5,
      classProbs = TRUE,
      summaryFunction = twoClassSummary
    )
  )
  predictors(rfe_fit)
}, error = function(e) all_vars)

vars_ga <- tryCatch({
  ga_ctrl <- gafsControl(
    functions = rfGA,
    method = "cv",
    number = 5,
    verbose = FALSE
  )

  ga_fit <- gafs(
    x = train[, all_vars, drop = FALSE],
    y = train[[target]],
    iters = 20,
    popSize = 30,
    gafsControl = ga_ctrl,
    metric = c(internal = "ROC", external = "ROC"),
    trControl = trainControl(
      method = "cv",
      number = 5,
      classProbs = TRUE,
      summaryFunction = twoClassSummary
    ),
    method = "rf"
  )

  ga_vars <- NULL
  if (!is.null(ga_fit$optVariables)) ga_vars <- ga_fit$optVariables
  if (is.null(ga_vars) && !is.null(ga_fit$ga$final)) ga_vars <- ga_fit$ga$final
  if (is.numeric(ga_vars)) ga_vars <- all_vars[ga_vars]

  unique(ga_vars)
}, error = function(e) all_vars)

feature_sets <- list(
  full = all_vars,
  forward = safe_vars(vars_forward, all_vars),
  backward = safe_vars(vars_backward, all_vars),
  stepwise = safe_vars(vars_stepwise, all_vars),
  rfe = safe_vars(vars_rfe, all_vars),
  genetic = safe_vars(vars_ga, all_vars)
)

feature_sets <- lapply(feature_sets, function(v) {
  if (length(v) < 2) all_vars else v
})

############################################################
# 9. XGBOOST OTIMIZADO
############################################################

run_xgb_cv <- function(train_df, target_name, vars, seed = 123) {
  set.seed(seed)

  x_train <- as.matrix(train_df[, vars, drop = FALSE])
  y_train <- ifelse(train_df[[target_name]] == "Yes", 1, 0)

  dtrain <- xgb.DMatrix(data = x_train, label = y_train)

  grid <- expand.grid(
    max_depth = c(3, 5, 7),
    eta = c(0.01, 0.05, 0.1),
    min_child_weight = c(1, 5),
    gamma = c(0, 1),
    subsample = c(0.8, 1.0),
    colsample_bytree = c(0.7, 0.9)
  )

  # Reduz combinacoes para manter tempo de execucao viavel
  grid <- grid %>% sample_n(size = min(24, nrow(grid)))

  best_auc <- -Inf
  best_params <- NULL
  best_nrounds <- 100

  for (i in seq_len(nrow(grid))) {
    params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = grid$max_depth[i],
      eta = grid$eta[i],
      min_child_weight = grid$min_child_weight[i],
      gamma = grid$gamma[i],
      subsample = grid$subsample[i],
      colsample_bytree = grid$colsample_bytree[i]
    )

    cv_fit <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = 600,
      nfold = 5,
      stratified = TRUE,
      early_stopping_rounds = 30,
      verbose = 0
    )

    auc_col <- if ("test_auc_mean" %in% colnames(cv_fit$evaluation_log)) "test_auc_mean" else "test_auc"
    best_iter <- cv_fit$best_iteration
    auc_now <- cv_fit$evaluation_log[best_iter, auc_col]

    if (!is.na(auc_now) && auc_now > best_auc) {
      best_auc <- auc_now
      best_params <- params
      best_nrounds <- best_iter
    }
  }

  final_model <- xgb.train(
    params = best_params,
    data = dtrain,
    nrounds = best_nrounds,
    watchlist = list(train = dtrain),
    verbose = 0
  )

  list(
    model = final_model,
    auc_cv = as.numeric(best_auc),
    vars = vars,
    params = best_params,
    nrounds = best_nrounds
  )
}

# testa cada estrategia de selecao e escolhe a melhor para XGBoost
set.seed(123)
xgb_candidates <- lapply(feature_sets, function(v) run_xgb_cv(train, target, v, seed = 123))

best_name <- names(which.max(sapply(xgb_candidates, function(m) m$auc_cv)))
best_xgb <- xgb_candidates[[best_name]]

cat("\nMelhor estrategia de selecao para XGBoost:", best_name, "\n")
cat("AUC CV:", round(best_xgb$auc_cv, 4), "\n")
cat("Numero de variaveis:", length(best_xgb$vars), "\n")

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
  nome = paste0("XGBoost (", best_name, ")"),
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
