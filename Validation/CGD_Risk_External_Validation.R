# CGD-Risk external validation

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

# Data
df_train <- read_csv("Data/CHARLS_train.csv") %>% select(score, age, income_total, be001, da049, ba000_w2_3,
                                                         dc028, da002, bd001_w2_4, da002_w2_1, da051_1_, da039, bb001_w3_2)
df_test <- read_csv("Data/CLHLS.csv") %>% select(-id)

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

paste0("CGD-Risk External Validation Set AUC: ", round(test_auc, round_to))
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
                scale_y_continuous(breaks = c(0, 300)) +
                theme_minimal() +
                labs(x = "Predicted Probability", y = "", title = "") 

calibration_plt <- ggarrange(calib_plt, predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))
calibration_plt

# Save if desired
#ggsave(filename = "CGD_Risk_Calibration_ext.tiff", width = 6, height = 4, units = "in", device = "tiff", dpi = 700)

# Net Benefit Plot
y <-  df_test$score
dca_df <- data.frame(y, cat_test_preds) %>% dplyr::rename(`CGD-Risk` = cat_test_preds)

dca_df <- dca(y ~ `CGD-Risk`, data = dca_df)$dca

net_benefit_plt <- dca_df %>%
                   dplyr::rename(`Screening Strategy` = label) %>%
                   mutate(`Screening Strategy` = plyr::revalue(`Screening Strategy`, c("Treat All" = "Refer All", "Treat None" = "Refer None"))) %>%
                   ggplot() +
                   geom_line(aes(x = threshold, y = net_benefit, 
                                 group = `Screening Strategy`, 
                                 color = `Screening Strategy`), size = 1.2) +
                   scale_color_manual(values = c(wes_palette("IsleofDogs1")[1],
                                                 wes_palette("GrandBudapest1")[2],
                                                 wes_palette("Darjeeling1")[2])) +
                   scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                   ylim(-0.05, 0.3) +
                   theme_minimal() +
                   theme(plot.title = element_text(size = 10, face = "bold")) +
                   labs(x = "Probability Threshold", y = "Net Benefit", title = "")

net_benefit_plt

# Save if desired
#ggsave(filename = "CGD_Risk_Net_Benefit_ext.tiff", width = 6, height = 4, units = "in", device = "tiff", dpi = 700)

# Net benefit table
cat_dca_df <- dca(y ~ cat_test_preds, data = calib_df)$dca

dca_table <- cat_dca_df %>% select(label, threshold, net_benefit) %>% 
             dplyr::rename(`Screening Strategy` = label) %>% 
             mutate(`Screening Strategy` = plyr::revalue(`Screening Strategy`, c("Treat All" = "Refer All", "Treat None" = "Refer None", "cat_test_preds" = "CGD-Risk")),
                     net_benefit = round(net_benefit, 3)) %>%
            filter(threshold %in% c(0.000000001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.71, 0.8, 0.99)) %>%
            pivot_wider(names_from = threshold, values_from = net_benefit)

# Table using gt
dca_table %>% gt(rowname_col = "Screening Strategy") %>%
              tab_header(title = "Net Benefit at Various Risk Thresholds") %>%
              tab_spanner(label = "Risk Threshold", 
                          columns = c("1e-09", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.71", "0.8", "0.99")) %>%
              tab_options(table.width = 800,
                          table.border.top.color = "black",
                          column_labels.border.bottom.color = "black",
                          column_labels.border.bottom.width= px(3)) 