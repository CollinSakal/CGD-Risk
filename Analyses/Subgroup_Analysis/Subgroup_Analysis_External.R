# CGD-Risk subgroup analysis using the external validation data

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
library(gt)
library(gtable)

# Seed
seed <- 19970507
set.seed(seed)

# Initialize value
round_to <- 2        # Round all output metrics (AUC, etc) to this many places following the decimal

##########################################################################################################################
# Rural only
##########################################################################################################################

# Data
df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3, dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039, bb001_w3_2)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(bb001_w3_2 == 3)

# Recipes
data_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 

# Parameters
params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
               iterations = 160, depth = 2, learning_rate = 0.07, verbose = 100)

# Get outcomes
y_train <- df_train$score
y_test <- df_test$score

# Preprocess training and testing sets
prepped_recipe <- prep(data_recipe, training = df_train, fresh = TRUE)

df_train_imp <- bake(prepped_recipe, new_data = df_train)
df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

df_test_imp <- bake(prepped_recipe, new_data = df_test)
df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

# Train and predict
cgd_train <- catboost.train(learn_pool = df_train_imp, params = params)
cgd_test_preds <- catboost.predict(model = cgd_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cgd_test_preds)

paste0("CGD-Risk C-statistic: ", round(test_auc, round_to))
ci.auc(y_test, cgd_test_preds)

# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cgd_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cgd_test_preds, data = calib_df, class = "1")$data %>% 
                mutate(midpoint = midpoint/100, 
                       Percent = Percent/100,
                       Lower = Lower/100,
                       Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt <- calib_plt_df %>% 
             ggplot() +
             geom_abline(linetype = "dashed", color = "black") +
             geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
             geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
             geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
             ylim(0,1) +
             theme_minimal() +
             theme(plot.title = element_text(size = 10, face = "bold")) +
             labs(x = "", y = "Observed Probability", title = "A")

predprob_plt <- calib_df %>%
                ggplot() +
                geom_histogram(aes(x = cgd_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt1 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cgd_test_preds) %>% dplyr::rename(`CGD-Risk` = cgd_test_preds)

dca_df <- dca(y ~ `CGD-Risk`, data = dca_df)$dca

net_benefit_plt1 <- dca_df %>%
                    dplyr::rename(`Screening Method` = label) %>%
                    mutate(`Screening Method` = plyr::revalue(`Screening Method`, c("Treat All" = "Refer All", "Treat None" = "Refer None"))) %>%
                    ggplot() +
                    geom_line(aes(x = threshold, y = net_benefit, 
                                  group = `Screening Method`, color = `Screening Method`), size = 1.2) +
                    scale_color_manual(values = c("green4",
                                                  "black",
                                                  wes_palette("GrandBudapest2")[1],
                                                  wes_palette("GrandBudapest2")[4],
                                                  wes_palette("Darjeeling1")[1],
                                                  wes_palette("Darjeeling1")[2],
                                                  wes_palette("Darjeeling2")[3])) +
                    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                    ylim(-0.05, 0.4) +
                    theme_minimal() +
                    theme(plot.title = element_text(size = 10, face = "bold")) +
                    labs(x = "Probability Threshold", y = "Net Benefit", title = "A")

# Feature importance
cat_FI_loss <- catboost.get_feature_importance(cgd_train, pool = df_test_imp, type = 'LossFunctionChange')

prednames <- c("Age",
               "Income",
               "Marital Status",
               "Hours of Sleep per Night",
               "Sex",
               "Satisfaction with Life",
               "Health Status",
               "Education",
               "Health Status vs Last Survey",
               "Intense Phsyical Activity >10 Minutes per Week",
               "Hearing Status",
               "Residence Location"
)

cat_FI_df <- data.frame(importance = cat_FI_loss[,1], predictor = prednames)

cat_FI_loss_plt1 <- cat_FI_df %>%
                    mutate(importance = as.numeric(importance)) %>%
                    mutate(predictor = factor(predictor)) %>%
                    mutate(posneg = ifelse(importance > 0, 1, 0)) %>%
                    mutate(posneg = factor(posneg)) %>%
                    ggplot() +
                    geom_col(aes(x = reorder(predictor, importance), y = importance, fill = posneg)) +
                    scale_fill_manual(values = c(wes_palette("Darjeeling1")[1], wes_palette("Darjeeling1")[2])) +
                    theme_minimal() +
                    coord_flip() +
                    theme(legend.position = 'none', plot.title = element_text(size = 10, face = "bold"),
                    text = element_text(size = 8)) +
                    labs(title = "A", x = "Predictor", y = "Difference in Loss")

##########################################################################################################################
# Urban only
##########################################################################################################################

# Data
df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3, dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039, bb001_w3_2)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(bb001_w3_2 == 1)

# Recipes
data_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 

# Parameters
params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
               iterations = 160, depth = 2, learning_rate = 0.07, verbose = 100)

# Get outcomes
y_train <- df_train$score
y_test <- df_test$score

# Preprocess training and testing sets
prepped_recipe <- prep(data_recipe, training = df_train, fresh = TRUE)

df_train_imp <- bake(prepped_recipe, new_data = df_train)
df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

df_test_imp <- bake(prepped_recipe, new_data = df_test)
df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

# Train and predict
cgd_train <- catboost.train(learn_pool = df_train_imp, params = params)
cgd_test_preds <- catboost.predict(model = cgd_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cgd_test_preds)

paste0("CGD-Risk C-statistic: ", round(test_auc, round_to))
ci.auc(y_test, cgd_test_preds)

# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cgd_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cgd_test_preds, data = calib_df, class = "1")$data %>% 
                mutate(midpoint = midpoint/100, 
                       Percent = Percent/100,
                       Lower = Lower/100,
                       Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt <- calib_plt_df %>% 
             ggplot() +
             geom_abline(linetype = "dashed", color = "black") +
             geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
             geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
             geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
             ylim(0,1) +
             theme_minimal() +
             theme(plot.title = element_text(size = 10, face = "bold")) +
             labs(x = "", y = "Observed Probability", title = "B")

predprob_plt <- calib_df %>%
                ggplot() +
                geom_histogram(aes(x = cgd_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt2 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cgd_test_preds) %>% dplyr::rename(`CGD-Risk` = cgd_test_preds)

dca_df <- dca(y ~ `CGD-Risk`, data = dca_df)$dca

net_benefit_plt2 <- dca_df %>%
                    dplyr::rename(`Screening Method` = label) %>%
                    mutate(`Screening Method` = plyr::revalue(`Screening Method`, c("Treat All" = "Refer All", "Treat None" = "Refer None"))) %>%
                    ggplot() +
                    geom_line(aes(x = threshold, y = net_benefit, 
                                  group = `Screening Method`, 
                                  color = `Screening Method`), size = 1.2) +
                    scale_color_manual(values = c("green4",
                                                  "black",
                                                  wes_palette("GrandBudapest2")[1],
                                                  wes_palette("GrandBudapest2")[4],
                                                  wes_palette("Darjeeling1")[1],
                                                  wes_palette("Darjeeling1")[2],
                                                  wes_palette("Darjeeling2")[3])) +
                    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                    ylim(-0.05, 0.4) +
                    theme_minimal() +
                    theme(plot.title = element_text(size = 10, face = "bold")) +
                    labs(x = "Probability Threshold", y = "Net Benefit", title = "B")

# Feature importance
cat_FI_loss <- catboost.get_feature_importance(cgd_train, pool = df_test_imp, type = 'LossFunctionChange')

prednames <- c("Age",
               "Income",
               "Marital Status",
               "Hours of Sleep per Night",
               "Sex",
               "Satisfaction with Life",
               "Health Status",
               "Education",
               "Health Status vs Last Survey",
               "Intense Phsyical Activity >10 Minutes per Week",
               "Hearing Status",
               "Residence Location"
)

cat_FI_df <- data.frame(importance = cat_FI_loss[,1], predictor = prednames)

cat_FI_loss_plt2 <- cat_FI_df %>%
                    mutate(importance = as.numeric(importance)) %>%
                    mutate(predictor = factor(predictor)) %>%
                    mutate(posneg = ifelse(importance > 0, 1, 0)) %>%
                    mutate(posneg = factor(posneg)) %>%
                    ggplot() +
                    geom_col(aes(x = reorder(predictor, importance), y = importance, fill = posneg)) +
                    scale_fill_manual(values = c(wes_palette("Darjeeling1")[1], wes_palette("Darjeeling1")[2])) +
                    theme_minimal() +
                    coord_flip() +
                    theme(legend.position = 'none', plot.title = element_text(size = 10, face = "bold"),
                          text = element_text(size = 8)) +
                    labs(title = "B", x = "Predictor", y = "Difference in Loss")

##########################################################################################################################
# Men only
##########################################################################################################################

# Data
df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3, dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039, bb001_w3_2)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(ba000_w2_3 == 1)

# Recipes
data_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 

# Parameters
params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
               iterations = 160, depth = 2, learning_rate = 0.07, verbose = 100)

# Get outcomes
y_train <- df_train$score
y_test <- df_test$score

# Preprocess training and testing sets
prepped_recipe <- prep(data_recipe, training = df_train, fresh = TRUE)

df_train_imp <- bake(prepped_recipe, new_data = df_train)
df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

df_test_imp <- bake(prepped_recipe, new_data = df_test)
df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

# Train and predict
cgd_train <- catboost.train(learn_pool = df_train_imp, params = params)
cgd_test_preds <- catboost.predict(model = cgd_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cgd_test_preds)

paste0("CGD-Risk C-statistic: ", round(test_auc, round_to))
ci.auc(y_test, cgd_test_preds)


# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cgd_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cgd_test_preds, data = calib_df, class = "1")$data %>% 
                mutate(midpoint = midpoint/100, 
                       Percent = Percent/100,
                       Lower = Lower/100,
                       Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt <- calib_plt_df %>% 
             ggplot() +
             geom_abline(linetype = "dashed", color = "black") +
             geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
             geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
             geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
             ylim(0,1) +
             theme_minimal() +
             theme(plot.title = element_text(size = 10, face = "bold")) +
             labs(x = "", y = "Observed Probability", title = "C")

predprob_plt <- calib_df %>%
                ggplot() +
                geom_histogram(aes(x = cgd_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt3 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cgd_test_preds) %>% dplyr::rename(`CGD-Risk` = cgd_test_preds)

dca_df <- dca(y ~ `CGD-Risk`, data = dca_df)$dca

net_benefit_plt3 <- dca_df %>%
                    dplyr::rename(`Screening Method` = label) %>%
                    mutate(`Screening Method` = plyr::revalue(`Screening Method`, c("Treat All" = "Refer All", "Treat None" = "Refer None"))) %>%
                    ggplot() +
                    geom_line(aes(x = threshold, y = net_benefit, 
                                  group = `Screening Method`, color = `Screening Method`), size = 1.2) +
                    scale_color_manual(values = c("green4",
                                                  "black",
                                                  wes_palette("GrandBudapest2")[1],
                                                  wes_palette("GrandBudapest2")[4],
                                                  wes_palette("Darjeeling1")[1],
                                                  wes_palette("Darjeeling1")[2],
                                                  wes_palette("Darjeeling2")[3])) +
                    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                    ylim(-0.05, 0.4) +
                    theme_minimal() +
                    theme(plot.title = element_text(size = 10, face = "bold")) +
                    labs(x = "Probability Threshold", y = "Net Benefit", title = "C")

# Feature Importance
cat_FI_loss <- catboost.get_feature_importance(cgd_train, pool = df_test_imp, type = 'LossFunctionChange')

prednames <- c("Age",
               "Income",
               "Marital Status",
               "Hours of Sleep per Night",
               "Sex",
               "Satisfaction with Life",
               "Health Status",
               "Education",
               "Health Status vs Last Survey",
               "Intense Phsyical Activity >10 Minutes per Week",
               "Hearing Status",
               "Residence Location"
)

cat_FI_df <- data.frame(importance = cat_FI_loss[,1], predictor = prednames)

cat_FI_loss_plt3 <- cat_FI_df %>%
                    mutate(importance = as.numeric(importance)) %>%
                    mutate(predictor = factor(predictor)) %>%
                    mutate(posneg = ifelse(importance > 0, 1, 0)) %>%
                    mutate(posneg = factor(posneg)) %>%
                    ggplot() +
                    geom_col(aes(x = reorder(predictor, importance), y = importance, fill = posneg)) +
                    scale_fill_manual(values = c(wes_palette("Darjeeling1")[1], wes_palette("Darjeeling1")[2])) +
                    theme_minimal() +
                    coord_flip() +
                    theme(legend.position = 'none', plot.title = element_text(size = 10, face = "bold"),
                          text = element_text(size = 8)) +
                    labs(title = "C", x = "Predictor", y = "Difference in Loss")

##########################################################################################################################
# Women only
##########################################################################################################################

# Data
df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3,dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039, bb001_w3_2)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(ba000_w2_3 == 0)

# Recipes
data_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 

# Parameters
params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
               iterations = 160, depth = 2, learning_rate = 0.07, verbose = 100)

# Get outcomes
y_train <- df_train$score
y_test <- df_test$score

# Preprocess training and testing sets
prepped_recipe <- prep(data_recipe, training = df_train, fresh = TRUE)

df_train_imp <- bake(prepped_recipe, new_data = df_train)
df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

df_test_imp <- bake(prepped_recipe, new_data = df_test)
df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

# Train and predict
cgd_train <- catboost.train(learn_pool = df_train_imp, params = params)
cgd_test_preds <- catboost.predict(model = cgd_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cgd_test_preds)

paste0("CGD-Risk C-statistic: ", round(test_auc, round_to))
ci.auc(y_test, cgd_test_preds)


# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cgd_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cgd_test_preds, data = calib_df, class = "1")$data %>% 
                mutate(midpoint = midpoint/100, 
                       Percent = Percent/100,
                       Lower = Lower/100,
                       Upper = Upper/100) %>% dplyr::rename(Proportion = Percent)

calib_plt <- calib_plt_df %>% 
             ggplot() +
             geom_abline(linetype = "dashed", color = "black") +
             geom_line(aes(x = midpoint, y = Proportion), color = wes_palette("Darjeeling1")[2], size = 1) +
             geom_errorbar(aes(x = midpoint, ymin = Lower, ymax = Upper), color = wes_palette("GrandBudapest1")[2], width = 0.03) +
             geom_point(aes(x = midpoint, y = Proportion), color = wes_palette("GrandBudapest1")[2], size = 2) +
             ylim(0,1) +
             theme_minimal() +
             theme(plot.title = element_text(size = 10, face = "bold")) +
             labs(x = "", y = "Observed Probability", title = "D")

predprob_plt <- calib_df %>%
                ggplot() +
                geom_histogram(aes(x = cgd_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt4 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cgd_test_preds) %>% dplyr::rename(`CGD-Risk` = cgd_test_preds)

dca_df <- dca(y ~ `CGD-Risk`, data = dca_df)$dca

net_benefit_plt4 <- dca_df %>%
                    dplyr::rename(`Screening Method` = label) %>%
                    mutate(`Screening Method` = plyr::revalue(`Screening Method`, c("Treat All" = "Refer All", "Treat None" = "Refer None"))) %>%
                    ggplot() +
                    geom_line(aes(x = threshold, y = net_benefit, 
                                  group = `Screening Method`, color = `Screening Method`), size = 1.2) +
                    scale_color_manual(values = c("green4",
                                                  "black",
                                                  wes_palette("GrandBudapest2")[1],
                                                  wes_palette("GrandBudapest2")[4],
                                                  wes_palette("Darjeeling1")[1],
                                                  wes_palette("Darjeeling1")[2],
                                                  wes_palette("Darjeeling2")[3])) +
                    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                    ylim(-0.05, 0.4) +
                    theme_minimal() +
                    theme(plot.title = element_text(size = 10, face = "bold")) +
                    labs(x = "Probability Threshold", y = "Net Benefit", title = "D")

# Feature Importance
cat_FI_loss <- catboost.get_feature_importance(cgd_train, pool = df_test_imp, type = 'LossFunctionChange')

prednames <- c("Age",
               "Income",
               "Marital Status",
               "Hours of Sleep per Night",
               "Sex",
               "Satisfaction with Life",
               "Health Status",
               "Education",
               "Health Status vs Last Survey",
               "Intense Phsyical Activity >10 Minutes per Week",
               "Hearing Status",
               "Residence Location"
)

cat_FI_df <- data.frame(importance = cat_FI_loss[,1], predictor = prednames)

cat_FI_loss_plt4 <- cat_FI_df %>%
                    mutate(importance = as.numeric(importance)) %>%
                    mutate(predictor = factor(predictor)) %>%
                    mutate(posneg = ifelse(importance > 0, 1, 0)) %>%
                    mutate(posneg = factor(posneg)) %>%
                    ggplot() +
                    geom_col(aes(x = reorder(predictor, importance), y = importance, fill = posneg)) +
                    scale_fill_manual(values = c(wes_palette("Darjeeling1")[1], wes_palette("Darjeeling1")[2])) +
                    theme_minimal() +
                    coord_flip() +
                    theme(legend.position = 'none', plot.title = element_text(size = 10, face = "bold"),
                          text = element_text(size = 8)) +
                    labs(title = "D", x = "Predictor", y = "Difference in Loss")

# Cobmined Plots
# Calibration
calib_combined <- ggarrange(calibration_plt1,
                            calibration_plt2,
                            calibration_plt3,
                            calibration_plt4,
                            nrow = 1,
                            ncol = 4)

calib_combined

ggsave(filename = "Subgroup_calib_ext.tiff", width = 7, height = 4, units = "in", device = "tiff", dpi = 1200)

# Net Benefit
net_benefit_combined <- ggarrange(net_benefit_plt1,
                                  net_benefit_plt2,
                                  net_benefit_plt3,
                                  net_benefit_plt4,
                                  nrow = 2,
                                  ncol = 2)

net_benefit_combined

ggsave(filename = "Subgroup_nb_ext.tiff", width = 7, height = 4.75, units = "in", device = "tiff", dpi = 1200)

# Feature importance combined
FI_combined <- ggarrange(cat_FI_loss_plt1,
                         cat_FI_loss_plt2,
                         cat_FI_loss_plt3,
                         cat_FI_loss_plt4,
                         nrow = 4,
                         ncol = 1)

FI_combined

ggsave(filename = "Subgroup_FI_ext.tiff", width = 7, height = 11, units = "in", device = "tiff", dpi = 1200)
