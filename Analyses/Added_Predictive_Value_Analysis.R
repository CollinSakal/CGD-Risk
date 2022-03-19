# Feature importance comparison by net benefit and calibration

# Libraries
library(catboost)
library(dplyr)
library(tidyr)
library(readr)
library(caret)
library(pROC)
library(ggpubr)
library(wesanderson)
library(dcurves)
library(recipes)
library(ggridges)
library(gtable)
library(gt)

# Seed
seed <- 19970507
set.seed(seed)

# Initialize values
numfolds <- 5        # Number of folds for K-fold cross-validation
round_to <- 2        # Round all output metrics (AUC, etc) to this many places following the decimal

# Data
df_train <- read_csv("Data/CHARLS_train.csv") 
df_test <- read_csv("Data/CHARLS_test.csv") 

##########################################################################################################################################################
# Demographics only
##########################################################################################################################################################
df_train_dems <- df_train %>% select(score, ba000_w2_3, age, income_total, bd001_w2_4)
df_test_dems <- df_test %>% select(score, ba000_w2_3, age, income_total, bd001_w2_4)

train_recipe_dems <- recipe(score ~., data = df_train_dems) %>% step_impute_knn(all_predictors())          
valid_recipe_dems <- recipe(score ~., data = df_train_dems) %>% step_impute_knn(all_predictors()) 
test_recipe_dems <- recipe(score ~., data = df_train_dems) %>% step_impute_knn(all_predictors()) 

params_dems <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                    iterations = 18, depth = 4, learning_rate = 0.1, verbose = 100)

# Get Outcomes
score <- df_test$score
y_test <- df_test$score
y_train <- df_train$score

# Set up and pre-process the training set
df_train_dems <- bake(prep(train_recipe_dems), new_data = df_train_dems)
df_train_dems <- catboost.load_pool(data = df_train_dems %>% select(-score), label = y_train) 

cat_model_dems <- catboost.train(learn_pool = df_train_dems, params = params_dems)

# Set up and pre-process the test set
df_test_dems <- bake(prep(test_recipe_dems), new_data = df_test_dems)
df_test_dems <- catboost.load_pool(data = df_test_dems %>% select(-score), label = y_test) 

# Predict on the test set
preds_test_dems <- catboost.predict(model = cat_model_dems, pool = df_test_dems, prediction_type = 'Probability')

# Get AUCs
paste0("Test set AUC for Demographics: ", round(auc(response = y_test, predictor = preds_test_dems), round_to))
round(ci.auc(y_test, preds_test_dems), round_to)

##########################################################################################################################################################
# Demographics + Satisfaction
##########################################################################################################################################################
score <- df_train$score

df_train_dems <- df_train %>% select(ID, ba000_w2_3, age, income_total, bd001_w2_4)

df_train_stsf <- df_train %>% select(ID, dc028, dc042_w3)
df_train_stsf <- inner_join(df_train_dems, df_train_stsf, by = "ID") %>% select(-ID) %>% mutate(score = score)

train_recipe_stsf <- recipe(score ~., data = df_train_stsf) %>% step_impute_knn(all_predictors())          
valid_recipe_stsf <- recipe(score ~., data = df_train_stsf) %>% step_impute_knn(all_predictors())
test_recipe_stsf <- recipe(score ~., data = df_train_stsf) %>% step_impute_knn(all_predictors()) 

params_stsf <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                    iterations = 86, depth = 2, learning_rate = 0.1, verbose = 100)

# Get Outcomes
score <- df_test$score
y_test <- df_test$score
y_train <- df_train$score

# Set up and pre-process the training set
df_train_stsf <- bake(prep(train_recipe_stsf), new_data = df_train_stsf)
df_train_stsf <- catboost.load_pool(data = df_train_stsf %>% select(-score), label = y_train) 

cat_model_stsf <- catboost.train(learn_pool = df_train_stsf, params = params_stsf)

# Set up and pre-process the test set
df_test_dems <- df_test %>% select(ID, ba000_w2_3, age, income_total, bd001_w2_4)

df_test_stsf <- df_test %>% select(ID, dc028, dc042_w3)
df_test_stsf <- inner_join(df_test_dems, df_test_stsf, by = "ID") %>% select(-ID) %>% mutate(score = score)

df_test_stsf <- bake(prep(test_recipe_stsf), new_data = df_test_stsf)
df_test_stsf <- catboost.load_pool(data = df_test_stsf %>% select(-score), label = y_test) 

# Predict on the test set
preds_test_stsf <- catboost.predict(model = cat_model_stsf, pool = df_test_stsf, prediction_type = 'Probability')

# Get AUCs
paste0("Test set AUC for Demographics + Satisfaction: ", round(auc(response = y_test, predictor = preds_test_stsf), round_to))
round(ci.auc(y_test, preds_test_stsf), round_to)

##########################################################################################################################################################
# Demographics + Satisfaction + Difficulties
##########################################################################################################################################################
score <- df_train$score

df_train_dems <- df_train %>% select(ID, ba000_w2_3, age, income_total, bd001_w2_4)
df_train_diff <- df_train %>% select(ID, diff_walking, diff_walking_km, db004, db005, db006, db007, db008, db009)
df_train_stsf <- df_train %>% select(ID, dc028, dc042_w3)

df_train_base <- inner_join(df_train_dems, df_train_stsf, by = "ID")

df_train_diff <- inner_join(df_train_base, df_train_diff, by = "ID") %>% select(-ID) %>% mutate(score = score)

train_recipe_diff <- recipe(score ~., data = df_train_diff) %>% step_impute_knn(all_predictors())          
valid_recipe_diff <- recipe(score ~., data = df_train_diff) %>% step_impute_knn(all_predictors()) 
test_recipe_diff <- recipe(score ~., data = df_train_diff) %>% step_impute_knn(all_predictors()) 

params_diff <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                    iterations = 69, depth = 4, learning_rate = 0.07, verbose = 100)

# Get Outcomes
score <- df_test$score
y_test <- df_test$score
y_train <- df_train$score

# Set up and pre-process the training set
df_train_diff <- bake(prep(train_recipe_diff), new_data = df_train_diff)
df_train_diff <- catboost.load_pool(data = df_train_diff %>% select(-score), label = y_train) 

cat_model_diff <- catboost.train(learn_pool = df_train_diff, params = params_diff)

# Set up and pre-process the test set
df_test_dems <- df_test %>% select(ID, ba000_w2_3, age, income_total, bd001_w2_4)
df_test_diff <- df_test %>% select(ID, diff_walking, diff_walking_km, db004, db005, db006, db007, db008, db009)
df_test_stsf <- df_test %>% select(ID, dc028, dc042_w3)

df_test_base <- inner_join(df_test_dems, df_test_stsf, by = "ID")

df_test_diff <- inner_join(df_test_base, df_test_diff, by = "ID") %>% select(-ID) %>% mutate(score = score)

df_test_diff <- bake(prep(test_recipe_diff), new_data = df_test_diff)
df_test_diff <- catboost.load_pool(data = df_test_diff %>% select(-score), label = y_test) 

# Predict on the test set
preds_test_diff <- catboost.predict(model = cat_model_diff, pool = df_test_diff, prediction_type = 'Probability')

# Get AUCs
paste0("Test set AUC for Demographics + Satisfaction + Difficulties: ", round(auc(response = y_test, predictor = preds_test_diff), round_to))
round(ci.auc(y_test, preds_test_diff), round_to)

##########################################################################################################################################################
# Demographics + Satisfaction + Difficulties + Health Status
##########################################################################################################################################################
score <- df_train$score

df_train_dems <- df_train %>% select(ID, ba000_w2_3, age, income_total, bd001_w2_4)
df_train_life <- df_train %>% select(ID, bb000_w3_1, bb000_w3_1, be001, da049, da051_1_)
df_train_diff <- df_train %>% select(ID, diff_walking, diff_walking_km, db004, db005, db006, db007, db008, db009)
df_train_stsf <- df_train %>% select(ID, dc028, dc042_w3)
df_train_hlth <- df_train %>% select(ID, da002, da002_w2_1, da034, da039, da041_w4)

df_train_base <- inner_join(df_train_dems, df_train_stsf, by = "ID")
df_train_base <- inner_join(df_train_base, df_train_diff, by = "ID")

df_train_hlth <- inner_join(df_train_base, df_train_hlth, by = "ID") %>% select(-ID) %>% mutate(score = score)

train_recipe_hlth <- recipe(score ~., data = df_train_hlth) %>% step_impute_knn(all_predictors())          
valid_recipe_hlth <- recipe(score ~., data = df_train_hlth) %>% step_impute_knn(all_predictors()) 
test_recipe_hlth <- recipe(score ~., data = df_train_hlth) %>% step_impute_knn(all_predictors())

params_hlth <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                    iterations = 224, depth = 2, learning_rate = 0.05, verbose = 100)

# Get Outcomes
score <- df_test$score
y_test <- df_test$score
y_train <- df_train$score

# Set up and pre-process the training set
df_train_hlth <- bake(prep(train_recipe_hlth), new_data = df_train_hlth)
df_train_hlth <- catboost.load_pool(data = df_train_hlth %>% select(-score), label = y_train) 

cat_model_hlth <- catboost.train(learn_pool = df_train_hlth, params = params_hlth)

# Set up and pre-process the test set
score <- df_test$score

df_test_dems <- df_test %>% select(ID, ba000_w2_3, age, income_total, bd001_w2_4)
df_test_life <- df_test %>% select(ID, bb000_w3_1, bb000_w3_1, be001, da049, da051_1_)
df_test_diff <- df_test %>% select(ID, diff_walking, diff_walking_km, db004, db005, db006, db007, db008, db009)
df_test_stsf <- df_test %>% select(ID, dc028, dc042_w3)
df_test_hlth <- df_test %>% select(ID, da002, da002_w2_1, da034, da039, da041_w4)

# New baseline is demographics + satisfaction + difficulties
df_test_base <- inner_join(df_test_dems, df_test_stsf, by = "ID")
df_test_base <- inner_join(df_test_base, df_test_diff, by = "ID")

# Create data frames: baseline and baseline + group for all groups
df_test_hlth <- inner_join(df_test_base, df_test_hlth, by = "ID") %>% select(-ID) %>% mutate(score = score)

df_test_hlth <- bake(prep(test_recipe_hlth), new_data = df_test_hlth)
df_test_hlth <- catboost.load_pool(data = df_test_hlth %>% select(-score), label = y_test) 

# Predict on the test set
preds_test_hlth <- catboost.predict(model = cat_model_hlth, pool = df_test_hlth, prediction_type = 'Probability')

# Get AUCs
paste0("Test set AUC for Demographics + Satisfaction + Difficulties + Health Status: ", 
       round(auc(response = y_test, predictor = preds_test_hlth), round_to))
round(ci.auc(y_test, preds_test_hlth), round_to)

##########################################################################################################################################################
# CGD-Risk Demographics + Satisfaction + Difficulties + Health Status + Lifestyle
##########################################################################################################################################################
df_train <- df_train %>% select(-ID)
df_test <- df_test %>% select(-ID)

train_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 
valid_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors())
test_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors())

cat_params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                   iterations = 200, depth = 2, learning_rate = 0.07, verbose = 100)

# Impute and train on the complete training set
df_train_imp <- bake(prep(train_recipe), new_data = df_train)
y_train <- df_train_imp$score

df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

cat_train <- catboost.train(learn_pool = df_train_imp, params = cat_params)

# Preprocess and get predictions for the test set
df_test_imp <- bake(prep(test_recipe), new_data = df_test)
y_test <- df_test_imp$score

df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

cat_test_preds <- catboost.predict(model = cat_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cat_test_preds)
test_auc_ci <- ci.auc(y_test, cat_test_preds)

paste0("CGD-Risk Test Set AUC: ", round(test_auc, round_to))
round(test_auc_ci,round_to)

##########################################################################################################################################################
# Net Benefit 
##########################################################################################################################################################
y <-  df_test$score
algs_dca_df <- data.frame(y, 
                          preds_test_dems, 
                          preds_test_stsf, 
                          preds_test_diff, 
                          preds_test_hlth, 
                          cat_test_preds) %>%
               dplyr::rename(`Demographics` = preds_test_dems,
                             `Demographics + Satisfaction` = preds_test_stsf,
                             `Demographics + Satisfaction + Difficulties` = preds_test_diff,
                             `Demographics + Satisfaction + Difficulties + Health Status` = preds_test_hlth,
                             `Demographics + Satisfaction + Difficulties + Health Status + Lifestyle (CGD-Risk)` = cat_test_preds)

algs_dca_df <- dca(y ~ `Demographics` +
                       `Demographics + Satisfaction` +
                       `Demographics + Satisfaction + Difficulties` +
                       `Demographics + Satisfaction + Difficulties + Health Status` +
                       `Demographics + Satisfaction + Difficulties + Health Status + Lifestyle (CGD-Risk)`, 
                   data = algs_dca_df)$dca

algs_net_benefit_plt <- algs_dca_df %>%
                        dplyr::rename(`Screening Strategy` = label) %>%
                        mutate(`Screening Strategy` = plyr::revalue(`Screening Strategy`, c("Treat All" = "Refer All", "Treat None" = "Refer None"))) %>%
                        ggplot() +
                        geom_line(aes(x = threshold, y = net_benefit, 
                                      group = `Screening Strategy`, color = `Screening Strategy`), size = 1.2) +
                        scale_color_manual(values = c("green4",
                                                      "black",
                                                      wes_palette("GrandBudapest2")[1],
                                                      wes_palette("GrandBudapest2")[4],
                                                      wes_palette("Darjeeling1")[1],
                                                      wes_palette("Darjeeling1")[2],
                                                      wes_palette("Darjeeling2")[3]
                        )) +
                        scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
                        ylim(-0.05, 0.4) +
                        theme_minimal() +
                        theme(plot.title = element_text(size = 10, face = "bold")) +
                        labs(x = "Probability Threshold", y = "Net Benefit", title = "")

algs_net_benefit_plt
#ggsave(filename = "FI_Net_Benefit3.tiff", width = 10.8, height = 5.54, units = "in", device = "tiff", dpi = 1200)

# Net Benefit comparison table
dca_table <- algs_dca_df %>% select(label, threshold, net_benefit) %>% 
             mutate(net_benefit = round(net_benefit, 2)) %>%
             dplyr::rename(`Screening Strategy` = label) %>% 
             mutate(`Screening Strategy` = plyr::revalue(`Screening Strategy`, c("Treat All" = "Refer All", 
                                                                             "Treat None" = "Refer None")),
                    net_benefit = round(net_benefit, 3)) %>%
             filter(threshold %in% c(0.000000001, 0.1, 0.2, 0.3, 0.4, 
                                     0.5, 0.6, 0.71, 0.8, 0.99)) %>%
             pivot_wider(names_from = threshold, values_from = net_benefit)

# Table using gt
dca_table %>% gt(rowname_col = "Screening Strategy") %>%
              tab_header(title = "Net Benefit at Various Probability Thresholds") %>%
              tab_spanner(label = "Probability Threshold", 
                          columns = c("1e-09", "0.1", "0.2", "0.3", "0.4", "0.5",
                                      "0.6", "0.71", "0.8", "0.99")) %>%
              tab_options(table.width = 800,
                          table.border.top.color = "black",
                          column_labels.border.bottom.color = "black",
                          column_labels.border.bottom.width= px(3)) 

##########################################################################################################################################################
# Calibration 
##########################################################################################################################################################

# Demographics only
y <-  df_test$score

calib_df_dems <- data.frame(y, preds_test_dems) %>% mutate(y = as.factor(y))
calib_plt_df_dems <- calibration(y ~ preds_test_dems, data = calib_df_dems, class = "1")$data %>% 
                     mutate(midpoint = midpoint/100, 
                            Percent = Percent/100,
                            Lower = Lower/100,
                            Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt_dems <- calib_plt_df_dems %>% 
                  ggplot() +
                  geom_abline(linetype = "dashed", color = "black") +
                  geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
                  geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
                  geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
                  ylim(0,1) +
                  theme_minimal() +
                  theme(plot.title = element_text(size = 10, face = "bold")) +
                  labs(x = "", y = "Observed Probability", title = "A")

predprob_plt_dems <- calib_df_dems %>%
                     ggplot() +
                     geom_histogram(aes(x = preds_test_dems), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                     scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
                     scale_y_continuous(breaks = c(0, 50, 100)) +
                     theme_minimal() +
                     labs(x = "Predicted Probability", y = "", title = "") 

calibration_dems <- ggarrange(calib_plt_dems, predprob_plt_dems, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_dems

# Demographics + Satisfaction
y <-  df_test$score

calib_df_stsf <- data.frame(y, preds_test_stsf) %>% mutate(y = as.factor(y))
calib_plt_df_stsf <- calibration(y ~ preds_test_stsf, data = calib_df_stsf, class = "1")$data %>% 
                     mutate(midpoint = midpoint/100, 
                            Percent = Percent/100,
                            Lower = Lower/100,
                            Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt_stsf <- calib_plt_df_stsf %>% 
                  ggplot() +
                  geom_abline(linetype = "dashed", color = "black") +
                  geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
                  geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
                  geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
                  ylim(0,1) +
                  theme_minimal() +
                  theme(plot.title = element_text(size = 10, face = "bold")) +
                  labs(x = "", y = "Observed Probability", title = "B")

predprob_plt_stsf <- calib_df_stsf %>%
                     ggplot() +
                     geom_histogram(aes(x = preds_test_stsf), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                     scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
                     scale_y_continuous(breaks = c(0, 50, 100)) +
                     theme_minimal() +
                     labs(x = "Predicted Probability", y = "", title = "") 

calibration_stsf <- ggarrange(calib_plt_stsf, predprob_plt_stsf, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_stsf

# Demographics + Satisfaction + Difficulties
y <-  df_test$score

calib_df_diff <- data.frame(y, preds_test_diff) %>% mutate(y = as.factor(y))
calib_plt_df_diff <- calibration(y ~ preds_test_diff, data = calib_df_diff, class = "1")$data %>% 
                     mutate(midpoint = midpoint/100, 
                            Percent = Percent/100,
                            Lower = Lower/100,
                            Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt_diff <- calib_plt_df_diff %>% 
                  ggplot() +
                  geom_abline(linetype = "dashed", color = "black") +
                  geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
                  geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
                  geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
                  ylim(0,1) +
                  theme_minimal() +
                  theme(plot.title = element_text(size = 10, face = "bold")) +
                  labs(x = "", y = "Observed Probability", title = "C")

predprob_plt_diff <- calib_df_diff %>%
                     ggplot() +
                     geom_histogram(aes(x = preds_test_diff), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                     scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
                     scale_y_continuous(breaks = c(0, 50, 100)) +
                     theme_minimal() +
                     labs(x = "Predicted Probability", y = "", title = "") 

calibration_diff <- ggarrange(calib_plt_diff, predprob_plt_diff, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_diff

# Demographics + Satisfaction + Difficulties + Health Status
y <-  df_test$score

calib_df_hlth <- data.frame(y, preds_test_hlth) %>% mutate(y = as.factor(y))
calib_plt_df_hlth <- calibration(y ~ preds_test_hlth, data = calib_df_hlth, class = "1")$data %>% 
                     mutate(midpoint = midpoint/100, 
                            Percent = Percent/100,
                            Lower = Lower/100,
                            Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt_hlth <- calib_plt_df_hlth %>% 
                  ggplot() +
                  geom_abline(linetype = "dashed", color = "black") +
                  geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
                  geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
                  geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
                  ylim(0,1) +
                  theme_minimal() +
                  theme(plot.title = element_text(size = 10, face = "bold")) +
                  labs(x = "", y = "Observed Probability", title = "D")

predprob_plt_hlth <- calib_df_hlth %>%
                     ggplot() +
                     geom_histogram(aes(x = preds_test_hlth), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                     scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
                     scale_y_continuous(breaks = c(0, 50, 100)) +
                     theme_minimal() +
                     labs(x = "Predicted Probability", y = "", title = "") 

calibration_hlth <- ggarrange(calib_plt_hlth, predprob_plt_hlth, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_hlth

# Demographics + Satisfaction + Difficulties + Health Status + Lifestyle (CGD-Risk)
y <-  df_test$score
preds_test_cgdr <- cat_test_preds

calib_df_cgdr <- data.frame(y, preds_test_cgdr) %>% mutate(y = as.factor(y))
calib_plt_df_cgdr <- calibration(y ~ preds_test_cgdr, data = calib_df_cgdr, class = "1")$data %>% 
                     mutate(midpoint = midpoint/100, 
                            Percent = Percent/100,
                            Lower = Lower/100,
                            Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt_cgdr <- calib_plt_df_cgdr %>% 
                  ggplot() +
                  geom_abline(linetype = "dashed", color = "black") +
                  geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
                  geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
                  geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
                  ylim(0,1) +
                  theme_minimal() +
                  theme(plot.title = element_text(size = 10, face = "bold")) +
                  labs(x = "", y = "Observed Probability", title = "E")

predprob_plt_cgdr <- calib_df_cgdr %>%
                     ggplot() +
                     geom_histogram(aes(x = preds_test_cgdr), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                     scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
                     scale_y_continuous(breaks = c(0, 50, 100)) +
                     theme_minimal() +
                     labs(x = "Predicted Probability", y = "", title = "") 

calibration_cgdr <- ggarrange(calib_plt_cgdr, predprob_plt_cgdr, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_cgdr

# Combined calibration plot
ggarrange(calib_plt_dems,
          calib_plt_stsf,
          calib_plt_diff,
          calib_plt_hlth,
          calib_plt_cgdr, nrow = 2, ncol = 3)

ggsave(filename = "FI_calibration_combined.tiff", width = 7, height = 5, units = "in", device = "tiff", dpi = 1600)

 


