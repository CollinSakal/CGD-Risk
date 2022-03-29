# Libraries
library(catboost)
library(tidyverse)
library(dplyr)
library(matrixStats)
library(caret)
library(pROC)
library(ggpubr)
library(recipes)

# Seed
seed <- 19970507
set.seed(seed)

# Initialize values
numfolds <- 5 # Number of folds for K-fold cross-validation
round_to <- 2 # Round all output metrics (AUC, etc) to this many places following the decimal

# Variable names for predictor groups
dem_vars <- c("ba000_w2_3", "age", "income_total", "bd001_w2_4")
stsf_vars <- c("dc028", "dc042_w3")
diff_vars <- c("diff_walking", "diff_walking_km", "db004", "db005", "db006", "db007", "db008", "db009")
hlth_vars <- c("da002", "da002_w2_1", "da034", "da039", "da041_w4")
life_vars <- c("bb000_w3_1", "bb001_w3_2", "be001", "da049", "da051_1_")

# Codes for filtering
#  - Rural: bb001_w3_2 == 3
#  - Urbanites: bb001_w3_2 == 1
#  - Men: ba000_w2_3 == 1
#  - Women: ba000_w2_3 == 0

# File paths 
CHARLS_train_path <- "Data/CHARLS_train.csv"
CHARLS_test_path <- "Data/CHARLS_test.csv"
CLHLS_path <- "Data/CLHLS.csv"
Sensitivity_path <- "Data/CHARLS_sensitivity"

# Data - vary the path, select(), and filter() depending on the analysis
df_train_full <- read_csv(CHARLS_train_path) 
df_test_full <- read_csv(CHARLS_test_path)

df_train <- df_train_full %>% 
            select(score, 
                   all_of(dem_vars), all_of(stsf_vars), 
                   all_of(diff_vars), all_of(life_vars),
                   all_of(hlth_vars)) 

df_test <- df_test_full %>% 
           filter(bb001_w3_2 == 1) %>% 
           select(score, 
                  all_of(dem_vars), all_of(stsf_vars), 
                  all_of(diff_vars), all_of(life_vars),
                  all_of(hlth_vars)) 

data_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 

# Hyperparameter tuning 
folds <- createFolds(y = df_train$score, k = numfolds)

param_grid <- expand.grid(loss_function = 'Logloss',            
                          custom_loss = 'AUC',                   
                          od_type = 'Iter',                     
                          od_wait = 100,                        
                          eval_metric = 'AUC',                  
                          use_best_model = TRUE,                
                          iterations = 10000,                    
                          depth = c(2,3,4,5,6),                 
                          learning_rate = c(0.005,0.015,0.03,0.05,0.07,0.1),
                          border_count = 254,                    
                          verbose = 100)

# Initialize matrices to store tuning metrics
iterations <- matrix(nrow = nrow(param_grid), ncol = numfolds)
tfold_aucs <- matrix(nrow = nrow(param_grid), ncol = numfolds)
vfold_aucs <- matrix(nrow = nrow(param_grid), ncol = numfolds)

for(j in 1:numfolds){
  
  train_fold <- df_train[-folds[[j]],]
  valid_fold <- df_train[folds[[j]],]
  
  y_train <- train_fold$score 
  y_valid <- valid_fold$score 
  
  prepped_recipe <- prep(data_recipe, training = train_fold, fresh = TRUE)
  
  train_fold <- bake(prepped_recipe, new_data = train_fold) 
  valid_fold <- bake(prepped_recipe, new_data = valid_fold)
  
  train_fold <- catboost.load_pool(data = train_fold %>% select(-score), label = y_train) 
  valid_fold <- catboost.load_pool(data = valid_fold %>% select(-score), label = y_valid)
  
  for(i in 1:nrow(param_grid)){
    
    params <-  as.list(param_grid[i,])
    
    cat_model <- catboost.train(learn_pool = train_fold, test_pool = valid_fold, params = params)
    
    preds_train <- catboost.predict(model = cat_model, pool = train_fold, prediction_type = 'Probability')
    preds_valid <- catboost.predict(model = cat_model, pool = valid_fold, prediction_type = 'Probability')
    
    tfold_aucs[i,j] <- auc(response = y_train, predictor = preds_train)
    vfold_aucs[i,j] <- auc(response = y_valid, predictor = preds_valid)
    iterations[i,j] <- cat_model$tree_count
    
  }
}

# Average CV information across the folds
param_grid$iterations <- ceiling(rowMeans(iterations))
param_grid$train_auc_mean <- rowMeans(tfold_aucs)
param_grid$valid_auc_mean <- rowMeans(vfold_aucs)
param_grid$train_auc_sd <- rowSds(tfold_aucs)
param_grid$valid_auc_sd <- rowSds(vfold_aucs)

# Order hyperparameters by average validation set AUC, select the best
param_grid <- param_grid[order(param_grid$valid_auc_mean, decreasing = TRUE),]

#params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
#               iterations = param_grid$iterations[1], 
#               depth = param_grid$depth[1], 
#               learning_rate = param_grid$learning_rate[1], 
#               border_count = 254, 
#               verbose = 100)

# Use these parameters for the "Full Model" 
params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
               iterations = 200, 
               depth = 2, 
               learning_rate = 0.07, 
               border_count = 254, 
               verbose = 100)

# Get Outcomes
y_test <- df_test$score
y_train <- df_train$score

# Set up and pre-process the whole training and testing sets
prepped_recipe <- prep(data_recipe, training = df_train, fresh = TRUE)

df_train <- bake(prepped_recipe, new_data = df_train)
df_train <- catboost.load_pool(data = df_train %>% select(-score), label = y_train) 

df_test <- bake(prepped_recipe, new_data = df_test)
df_test <- catboost.load_pool(data = df_test %>% select(-score), label = y_test) 

# Train and predict on the test set
cat_model <- catboost.train(learn_pool = df_train, params = params)
preds_test <- catboost.predict(model = cat_model, pool = df_test, prediction_type = 'Probability')

# Get AUC and 95% CI
paste0("Test set AUC: ", round(auc(response = y_test, predictor = preds_test), round_to))
round(ci.auc(y_test, preds_test), round_to)