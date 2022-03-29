# CGD-Risk Internal Validation

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

# Seed
seed <- 19970507
set.seed(seed)

# Initialize values
numfolds <- 5        # Number of folds for K-fold cross-validation
cv_repeats <- 200    # Number of cross-validation repeats
round_to <- 2        # Round all output metrics (AUC, etc) to this many places following the decimal

# Data
df_train <- read_csv("Data/CHARLS_train.csv") %>% select(-ID)
df_test <- read_csv("Data/CHARLS_test.csv") %>% select(-ID)

# Recipes
data_recipe <- recipe(score ~., data = df_train) %>% step_impute_knn(all_predictors()) 

# Parameters
cat_params <- list(loss_function = 'Logloss', custom_loss = 'AUC', eval_metric = 'AUC', 
                   iterations = 200, depth = 2, learning_rate = 0.07, verbose = 100)

# Repeated 5-fold Cross Validation
tfold_aucs <- c()
vfold_aucs <- c()

for(i in 1:cv_repeats){
  
  folds <- createFolds(y = df_train$score, k = numfolds)
  
  for(j in 1:numfolds){
    
    train_fold <- df_train[-folds[[j]], ]
    valid_fold <- df_train[folds[[j]], ]
    
    y_train <- train_fold$score 
    y_valid <- valid_fold$score 
    
    prepped_recipe <- prep(data_recipe, training = train_fold, fresh = TRUE)
    
    train_fold <- bake(prepped_recipe, new_data = train_fold) 
    valid_fold <- bake(prepped_recipe, new_data = valid_fold) 
    
    train_fold <- catboost.load_pool(data = train_fold %>% select(-score), label = y_train)
    valid_fold <- catboost.load_pool(data = valid_fold %>% select(-score), label = y_valid)
  
    cat_model <- catboost.train(learn_pool = train_fold, params = cat_params)
    
    tfold_preds <- catboost.predict(model = cat_model, pool = train_fold, prediction_type = 'Probability')
    vfold_preds <- catboost.predict(model = cat_model, pool = valid_fold, prediction_type = 'Probability')
    
    tfold_aucs <- c(tfold_aucs, pROC::auc(response = y_train, predictor = tfold_preds))
    vfold_aucs <- c(vfold_aucs, pROC::auc(response = y_valid, predictor = vfold_preds))
    
  }
}

auc_df <- data.frame(tfold_aucs, vfold_aucs)

tfold_aucs <- auc_df$tfold_aucs
vfold_aucs <- auc_df$vfold_aucs

# Sort AUCs and get confidence intervals 
tfold_aucs <- sort(tfold_aucs, decreasing = TRUE)
vfold_aucs <- sort(vfold_aucs, decreasing = TRUE)

tfold_auc_cil <- quantile(tfold_aucs, probs = 0.025); tfold_auc_point <- mean(tfold_aucs); tfold_auc_ciu <- quantile(tfold_aucs, probs = 0.975)
vfold_auc_cil <- quantile(vfold_aucs, probs = 0.025); vfold_auc_point <- mean(vfold_aucs); vfold_auc_ciu <- quantile(vfold_aucs, probs = 0.975) 

paste0("CGD-Risk Average Validation AUC: ", round(vfold_auc_point, round_to),
       " (", round(vfold_auc_cil, round_to), ", ", round(vfold_auc_ciu, round_to), ")")
paste0("CGD-Risk Average Training AUC: ", round(tfold_auc_point, round_to), 
       " (", round(tfold_auc_cil, round_to), ", ", round(tfold_auc_ciu, round_to), ")")

# Preprocess training and testing sets
prepped_recipe <- prep(data_recipe, training = df_train, fresh = TRUE)

y_train <- df_train$score
y_test <- df_test$score

df_train_imp <- bake(prepped_recipe, new_data = df_train)
df_train_imp <- catboost.load_pool(data = df_train_imp %>% select(-score), label = y_train)

df_test_imp <- bake(prepped_recipe, new_data = df_test)
df_test_imp <- catboost.load_pool(data = df_test_imp %>% select(-score), label = y_test)

# Train and predict
cat_train <- catboost.train(learn_pool = df_train_imp, params = cat_params)

cat_test_preds <- catboost.predict(model = cat_train, pool = df_test_imp, prediction_type = 'Probability')

# Calculate AUC
test_auc <- auc(response = y_test, predictor = cat_test_preds)

paste0("CGD-Risk Test Set AUC: ", round(test_auc, round_to))
ci.auc(y_test, cat_test_preds)

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
                 labs(x = "", y = "Observed Probability", title = "CGD-Risk Calibration")

cat_predprob_plt <- cat_calib_df %>%
                    ggplot() +
                    geom_histogram(aes(x = cat_test_preds), fill = wes_palette("IsleofDogs1")[1],bins = 200) +
                    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                    scale_y_continuous(breaks = c(0, 10)) +
                    theme_minimal() +
                    labs(x = "Predicted Probability", y = "", title = "") 

cat_calibration <- ggarrange(cat_calib_plt, cat_predprob_plt, nrow = 2, ncol = 1, heights = c(1, 0.30))
cat_calibration

# Save image if desired
#ggsave(filename = "CGD_Risk_Calibration_int.tiff", width = 7, height = 6, units = "in", device = "tiff", dpi = 700)

# Net Benefit
y <-  df_test$score

cat_dca_df <- dca(y ~ cat_test_preds, data = cat_calib_df)$dca
cat_dca_df$label <- as.numeric(cat_dca_df$label)

cat_net_benefit_plt <- cat_dca_df %>%
                       dplyr::rename(`Screening Strategy` = label) %>%
                       mutate(`Screening Strategy` = factor(`Screening Strategy`, levels = c(1,2,3), labels = c("Refer All", "Refer None", "CGD-Risk"))) %>%
                       ggplot() +
                       geom_line(aes(x = threshold, y = net_benefit, 
                                     group = `Screening Strategy`, 
                                     color = `Screening Strategy`), size = 1.2) +
                       scale_color_manual(values = c(wes_palette("IsleofDogs1")[1],
                                                     wes_palette("GrandBudapest1")[2],
                                                     wes_palette("Darjeeling1")[2])) +
                       scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
                       ylim(-0.05, 0.4) +
                       theme_minimal() +
                       theme(plot.title = element_text(size = 10, face = "bold")) +
                       labs(x = "Probability Threshold", y = "Net Benefit", title = "CGD-Risk Net Benefit Curve")

cat_net_benefit_plt


# Feature (predictor) importance plot
cat_PI_loss <- catboost.get_feature_importance(cat_train, pool = df_test_imp, type = 'LossFunctionChange')

prednames <- c("Sex",
               "Education",
               "Residence Type",
               "Residence Location",
               "Marital Status",
               "Health Status",
               "Health Status vs 3 Years Ago",
               "Hours of Sleep per Night",
               "Difficulty Getting up from a Chair",
               "Difficulty Climbing Multiple Flights of Stairs",
               "Difficulty Kneeling, Crouching, Stooping",
               "Difficulty Extending Arms Above Shoulders",
               "Difficulty Carrying Weights >10 Jin",
               "Difficulty Picking up a Small Coin",
               "Eyesight up Close",
               "Hearing Status",
               "Trouble with Body Pain",
               "Intense Phsyical Activity >10 Minutes per Week",
               "Satisfaction with Life",
               "Satisfaction with Health",
               "Age",
               "Difficulty Walking 100m",
               "Difficulty Walking 1km",
               "Income"
)

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
                   labs(title = "", x = "Predictor", y = "Difference in Loss")

cat_PI_loss_plt

