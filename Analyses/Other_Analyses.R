# This script allows the re-creation of all added predictive value and sub-group analyses

# Libraries
library(catboost)
library(tidyverse)
library(dplyr)
library(matrixStats)
library(caret)
library(pROC)
library(ggpubr)
library(recipes)
library(wesanderson)
library(dcurves)

# Seed
seed <- 19970507
set.seed(seed)

# Initialize values
numfolds <- 5 # Number of folds for K-fold cross-validation
round_to <- 2 # Round all output metrics (AUC, etc) to this many places following the decimal

# Variable names for predictor groups
dem_vars <- c("ba000_w2_3", "age", "income_total", "bd001_w2_4") # Demographic group
stsf_vars <- c("dc028", "dc042_w3") # Satisfaction group
diff_vars <- c("diff_walking", "diff_walking_km", "db004", "db005", "db006", "db007", "db008", "db009") # Activity difficulties group
hlth_vars <- c("da002", "da002_w2_1", "da034", "da039", "da041_w4") # Health status group
life_vars <- c("bb000_w3_1", "bb001_w3_2", "be001", "da049", "da051_1_") # Lifestyle group

# File paths (Must have run the "Cohort Creation" scripts first)
CHARLS_train_path <- "Data/CHARLS_train.csv"
CHARLS_test_path <- "Data/CHARLS_test.csv"
CLHLS_path <- "Data/CLHLS.csv"
Sensitivity_path <- "Data/CHARLS_sensitivity.csv"

# Arguments for filter() When doing the sub-group analyses
#  - Rural: bb001_w3_2 == 3
#  - Urbanites: bb001_w3_2 == 1
#  - Men: ba000_w2_3 == 1
#  - Women: ba000_w2_3 == 0

# Data - vary the arguments in read_csv(), select(), and filter() depending on the analysis
df_train_full <- read_csv(CHARLS_train_path) 
df_test_full <- read_csv(Sensitivity_path)

df_train <- df_train_full %>% 
            select(score, 
                   all_of(dem_vars), all_of(stsf_vars), all_of(hlth_vars), all_of(life_vars), all_of(diff_vars)) 

df_test <- df_test_full %>% 
           #filter(bb001_w3_2 == 1) %>% # Comment this line out if not doing a sub-group analysis
           select(score, 
                  all_of(dem_vars), all_of(stsf_vars), all_of(hlth_vars), all_of(life_vars), all_of(diff_vars)) 

data_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 

# Hyperparameter tuning (if doing an analysis for the full model, skip to line 127)
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
#               iterations = param_grid$iterations[1], depth = param_grid$depth[1], learning_rate = param_grid$learning_rate[1], 
#               border_count = 254, verbose = 100)

# Use these parameters for recreating any "Full Model" analyses 
params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
               iterations = 200, depth = 2, learning_rate = 0.07, border_count = 254, verbose = 100)

# Get Outcomes
y_test <- df_test$score
y_train <- df_train$score

# Set up and pre-process the whole training and testing sets
prepped_recipe <- prep(data_recipe, training = df_train, fresh = TRUE)

df_train_imp <- bake(prepped_recipe, new_data = df_train)
df_train_imp <- catboost.load_pool(data = df_train %>% select(-score), label = y_train) 

df_test_imp <- bake(prepped_recipe, new_data = df_test)
df_test_imp <- catboost.load_pool(data = df_test %>% select(-score), label = y_test) 

# Train and predict on the test set
cat_model <- catboost.train(learn_pool = df_train_imp, params = params)
cat_test_preds <- catboost.predict(model = cat_model, pool = df_test_imp, prediction_type = 'Probability')

# Get AUC and 95% CI
paste0("Test set AUC: ", round(auc(response = y_test, predictor = cat_test_preds), round_to))
round(ci.auc(y_test, cat_test_preds), round_to)

# Calibration
y <-  df_test$score

cat_calib_df <- data.frame(y, cat_test_preds) %>% mutate(y = as.factor(y))
cat_calib_plt_df <- calibration(y ~ cat_test_preds, data = cat_calib_df, class = "1")$data %>% 
                    mutate(midpoint = midpoint/100, 
                           Percent = Percent/100,
                           Lower = Lower/100,
                           Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

cat_calib_plt <- cat_calib_plt_df %>% 
                 ggplot() +
                 geom_abline(linetype = "dashed", color = "black") +
                 geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
                 geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
                 geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
                 theme_minimal() +
                 theme(plot.title = element_text(size = 10, face = "bold")) +
                 labs(x = "", y = "Observed Probability", title = "Calibration Plot")

cat_predprob_plt <- cat_calib_df %>%
                    ggplot() +
                    geom_histogram(aes(x = cat_test_preds), fill = wes_palette("IsleofDogs1")[1], bins = 200) +
                    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                    scale_y_continuous(breaks = c(0, 10)) +
                    theme_minimal() +
                    labs(x = "Predicted Probability", y = "", title = "") 

cat_calibration <- ggarrange(cat_calib_plt, cat_predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))
cat_calibration

# Net Benefit
cat_PI_loss <- catboost.get_feature_importance(cat_train, pool = df_test_imp, type = 'LossFunctionChange')

# Use these names for the CHARLS data
prednames <- c("Sex","Education","Residence Type","Residence Location","Marital Status","Health Status","Health Status vs 3 Years Ago",
               "Hours of Sleep per Night","Difficulty Getting up from a Chair","Difficulty Climbing Multiple Flights of Stairs","Difficulty Kneeling, Crouching, Stooping",
               "Difficulty Extending Arms Above Shoulders","Difficulty Carrying Weights >10 Jin","Difficulty Picking up a Small Coin",
               "Eyesight up Close","Hearing Status","Trouble with Body Pain","Intense Phsyical Activity >10 Minutes per Week",
               "Satisfaction with Life","Satisfaction with Health","Age","Difficulty Walking 100m","Difficulty Walking 1km","Income"
)

# Use these names for the CLHLS data
#prednames <- c("Age","Income","Marital Status","Hours of Sleep per Night","Sex","Satisfaction with Life","Health Status",
#               "Education","Health Status vs Last Survey","Intense Phsyical Activity >10 Minutes per Week","Hearing Status","Residence Location"
#)

cat_PI_df <- data.frame(importance = cat_PI_loss[,1], predictor = prednames)

cat_PI_loss_plt <- cat_PI_df %>%
                   mutate(importance = as.numeric(importance)) %>%
                   mutate(predictor = factor(predictor)) %>%
                   mutate(posneg = ifelse(importance > 0, 1, 0)) %>%
                   mutate(posneg = factor(posneg)) %>%
                   ggplot() +
                   geom_col(aes(x = reorder(predictor, importance), y = importance, fill = posneg)) +
                   scale_fill_manual(values = c(wes_palette("Darjeeling1")[1], wes_palette("Darjeeling1")[2])) +
                   theme_minimal() +
                   coord_flip() +
                   theme(legend.position = "none") + 
                   labs(title = "Net Benefit Curve", x = "Predictor", y = "Difference in Loss")

cat_PI_loss_plt