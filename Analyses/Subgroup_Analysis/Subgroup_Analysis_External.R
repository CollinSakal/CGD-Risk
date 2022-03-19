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

# Initialize values
numfolds <- 5        # Number of folds for K-fold cross-validation
cv_repeats <- 200    # Number of cross-validation repeats
round_to <- 2        # Round all output metrics (AUC, etc) to this many places following the decimal

##########################################################################################################################
# Rural only
##########################################################################################################################

# Data
df_train <- df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3,
                                                                     dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(bb001_w3_2 == 3)

# Recipes
train_recipe <- test_recipe <- recipe(score ~., data = df_train) %>%
                               step_impute_knn(all_predictors()) 

# Parameters
cat_params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                   iterations = 200, depth = 2, learning_rate = 0.07, verbose = 100)

# Impute and train on the complete training set
df_train_imp <- bake(prep(train_recipe), new_data = df_train)
y_train <- df_train_imp$score

df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

cat_train <- catboost.train(learn_pool = df_train_imp, params = cat_params)

# Preprocess and get predictions for the test set (CLHLS)
df_test_imp <- bake(prep(test_recipe), new_data = df_test)
y_test <- df_test_imp$score

df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

cat_test_preds <- catboost.predict(model = cat_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cat_test_preds)
test_auc_ci <- ci.auc(y_test, cat_test_preds)

paste0("CGD-Risk Sub-group Analysis (Rural) AUC: ", round(test_auc, round_to))
test_auc_ci

# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cat_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cat_test_preds, data = calib_df, class = "1")$data %>% 
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
                geom_histogram(aes(x = cat_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt1 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_plt1

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cat_test_preds) %>% dplyr::rename(`CGD-Risk` = cat_test_preds)

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

net_benefit_plt1

##########################################################################################################################
# Urban only
##########################################################################################################################

# Data
df_train <- df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3,
                                                                     dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(bb001_w3_2 == 1)

# Recipes
train_recipe <- test_recipe <- recipe(score ~., data = df_train) %>%
                               step_impute_knn(all_predictors()) 

# Parameters
cat_params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                   iterations = 200, depth = 2, learning_rate = 0.07, verbose = 100)

# Impute and train on the complete training set
df_train_imp <- bake(prep(train_recipe), new_data = df_train)
y_train <- df_train_imp$score

df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

cat_train <- catboost.train(learn_pool = df_train_imp, params = cat_params)

# Preprocess and get predictions for the test set (CLHLS)
df_test_imp <- bake(prep(test_recipe), new_data = df_test)
y_test <- df_test_imp$score

df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

cat_test_preds <- catboost.predict(model = cat_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cat_test_preds)
test_auc_ci <- ci.auc(y_test, cat_test_preds)

paste0("CGD-Risk Sub-group Analysis (Urban) AUC: ", round(test_auc, round_to))
test_auc_ci

# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cat_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cat_test_preds, data = calib_df, class = "1")$data %>% 
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
                geom_histogram(aes(x = cat_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt2 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_plt2

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cat_test_preds) %>% dplyr::rename(`CGD-Risk` = cat_test_preds)

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

net_benefit_plt2

##########################################################################################################################
# Men only
##########################################################################################################################

# Data
df_train <- df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3,
                                                                     dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(ba000_w2_3 == 1)

# Recipes
train_recipe <- test_recipe <- recipe(score ~., data = df_train) %>%
                               step_impute_knn(all_predictors()) 

# Parameters
cat_params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                   iterations = 200, depth = 2, learning_rate = 0.07, verbose = 100)

# Impute and train on the complete training set
df_train_imp <- bake(prep(train_recipe), new_data = df_train)
y_train <- df_train_imp$score

df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

cat_train <- catboost.train(learn_pool = df_train_imp, params = cat_params)

# Preprocess and get predictions for the test set (CLHLS)
df_test_imp <- bake(prep(test_recipe), new_data = df_test)
y_test <- df_test_imp$score

df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

cat_test_preds <- catboost.predict(model = cat_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cat_test_preds)
test_auc_ci <- ci.auc(y_test, cat_test_preds)

paste0("CGD-Risk Sub-group Analysis (Men) AUC: ", round(test_auc, round_to))
test_auc_ci

# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cat_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cat_test_preds, data = calib_df, class = "1")$data %>% 
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
                geom_histogram(aes(x = cat_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt3 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_plt3

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cat_test_preds) %>% dplyr::rename(`CGD-Risk` = cat_test_preds)

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

net_benefit_plt3

##########################################################################################################################
# Women only
##########################################################################################################################

# Data
df_train <- df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3,
                                                                     dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id) %>% filter(ba000_w2_3 == 0)

# Recipes
train_recipe <- test_recipe <- recipe(score ~., data = df_train) %>%
                               step_impute_knn(all_predictors()) 

# Parameters
cat_params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                   iterations = 200, depth = 2, learning_rate = 0.07, verbose = 100)

# Impute and train on the complete training set
df_train_imp <- bake(prep(train_recipe), new_data = df_train)
y_train <- df_train_imp$score

df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

cat_train <- catboost.train(learn_pool = df_train_imp, params = cat_params)

# Preprocess and get predictions for the test set (CLHLS)
df_test_imp <- bake(prep(test_recipe), new_data = df_test)
y_test <- df_test_imp$score

df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

cat_test_preds <- catboost.predict(model = cat_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cat_test_preds)
test_auc_ci <- ci.auc(y_test, cat_test_preds)

paste0("CGD-Risk Sub-group Analysis (Women) AUC: ", round(test_auc, round_to))
test_auc_ci

# Calibration
y <-  df_test$score

calib_df <- data.frame(y, cat_test_preds) %>% mutate(y = as.factor(y))
calib_plt_df <- calibration(y ~ cat_test_preds, data = calib_df, class = "1")$data %>% 
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
                geom_histogram(aes(x = cat_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                scale_y_continuous(breaks = c(0, 200)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt4 <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_plt4

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cat_test_preds) %>% dplyr::rename(`CGD-Risk` = cat_test_preds)

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

net_benefit_plt4

# Cobmined Plots
# Calibration
calib_combined <- ggarrange(calibration_plt1,
                            calibration_plt2,
                            calibration_plt3,
                            calibration_plt4,
                            nrow = 1,
                            ncol = 4)

calib_combined

#ggsave(filename = "Subgroup_calib_ext.tiff", width = 7, height = 4, units = "in", device = "tiff", dpi = 1200)
# Net Benefit
net_benefit_combined <- ggarrange(net_benefit_plt1,
                                  net_benefit_plt2,
                                  net_benefit_plt3,
                                  net_benefit_plt4,
                                  nrow = 2,
                                  ncol = 2)

net_benefit_combined

#ggsave(filename = "Subgroup_nb_ext.tiff", width = 7, height = 4.75, units = "in", device = "tiff", dpi = 1200)
