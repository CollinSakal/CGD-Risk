# CLHLS Cohort Filtering
library(tidyverse)
library(haven)
library(caret)

# Reading in the file
df <- read_sav("Data/clhls_2018.sav")

# Intialize values
cesd_cutoff <- 10
age_cutoff <- 65

# Isolate variables needed for the CESD, exclude if "not able to answer" (8) or "Missing" (9)
cesd <- df %>% select(id, b31, b32, b33, b34, b35, b36, b37, b38, b39, b310a) %>% 
               filter(b31 < 8) %>%
               filter(b32 < 8) %>%
               filter(b33 < 8) %>%
               filter(b34 < 8) %>%
               filter(b35 < 8) %>%
               filter(b36 < 8) %>%
               filter(b37 < 8) %>%
               filter(b38 < 8) %>%
               filter(b39 < 8) %>%
               filter(b310a < 8) 

# Recoding procedure for the variables:
#   1. Group 4 (seldom) and 5 (never) into one category
#   2. Shift all positive questions down 1 to create a range of 0-3
#         - b35, b37, b310a
#   3. Shift down and recode the remaining variables (1,2,3,4) -> (3,2,1,0)
#   4. Sum the scores and binarize s.t. score >= 10 -> 1, else -> 0

# 1. Group 4 and 5 together into one category
cesd$b31 <- ifelse(cesd$b31 == 5, 4, cesd$b31)
cesd$b32 <- ifelse(cesd$b32 == 5, 4, cesd$b32)
cesd$b33 <- ifelse(cesd$b33 == 5, 4, cesd$b33)
cesd$b34 <- ifelse(cesd$b34 == 5, 4, cesd$b34)
cesd$b35 <- ifelse(cesd$b35 == 5, 4, cesd$b35)
cesd$b36 <- ifelse(cesd$b36 == 5, 4, cesd$b36)
cesd$b37 <- ifelse(cesd$b37 == 5, 4, cesd$b37)
cesd$b38 <- ifelse(cesd$b38 == 5, 4, cesd$b38)
cesd$b39 <- ifelse(cesd$b39 == 5, 4, cesd$b39)
cesd$b310a <- ifelse(cesd$b310a == 5, 4, cesd$b310a)

# 2. Shift all positive questions down 1 to create a range of 0-3
cesd$b35 <- cesd$b35 - 1
cesd$b37 <-  cesd$b37 - 1
cesd$b310a <- cesd$b310a - 1

# 3. Shift and recode the remaining variables (1,2,3,4) -> (3,2,1,0)
cesd$b31 <- ifelse(cesd$b31 == 3, 100, cesd$b31) # Placeholder since 1s get replaced with 3s next
cesd$b31 <- ifelse(cesd$b31 == 1, 3, cesd$b31)
cesd$b31 <- ifelse(cesd$b31 == 100, 1, cesd$b31) # Now swap placeholder
cesd$b31 <- ifelse(cesd$b31 == 4, 0, cesd$b31)

cesd$b32 <- ifelse(cesd$b32 == 3, 100, cesd$b32) # Placeholder since 1s get replaced with 3s next
cesd$b32 <- ifelse(cesd$b32 == 1, 3, cesd$b32)
cesd$b32 <- ifelse(cesd$b32 == 100, 1, cesd$b32) # Now swap placeholder
cesd$b32 <- ifelse(cesd$b32 == 4, 0, cesd$b32)

cesd$b33 <- ifelse(cesd$b33 == 3, 100, cesd$b33) # Placeholder since 1s get replaced with 3s next
cesd$b33 <- ifelse(cesd$b33 == 1, 3, cesd$b33)
cesd$b33 <- ifelse(cesd$b33 == 100, 1, cesd$b33) # Now swap placeholder
cesd$b33 <- ifelse(cesd$b33 == 4, 0, cesd$b33)

cesd$b34 <- ifelse(cesd$b34 == 3, 100, cesd$b34) # Placeholder since 1s get replaced with 3s next
cesd$b34 <- ifelse(cesd$b34 == 1, 3, cesd$b34)
cesd$b34 <- ifelse(cesd$b34 == 100, 1, cesd$b34) # Now swap placeholder
cesd$b34 <- ifelse(cesd$b34 == 4, 0, cesd$b34)

cesd$b36 <- ifelse(cesd$b36 == 3, 100, cesd$b36) # Placeholder since 1s get replaced with 3s next
cesd$b36 <- ifelse(cesd$b36 == 1, 3, cesd$b36)
cesd$b36 <- ifelse(cesd$b36 == 100, 1, cesd$b36) # Now swap placeholder
cesd$b36 <- ifelse(cesd$b36 == 4, 0, cesd$b36)

cesd$b38 <- ifelse(cesd$b38 == 3, 100, cesd$b38) # Placeholder since 1s get replaced with 3s next
cesd$b38 <- ifelse(cesd$b38 == 1, 3, cesd$b38)
cesd$b38 <- ifelse(cesd$b38 == 100, 1, cesd$b38) # Now swap placeholder
cesd$b38 <- ifelse(cesd$b38 == 4, 0, cesd$b38)

cesd$b39 <- ifelse(cesd$b39 == 3, 100, cesd$b39) # Placeholder since 1s get replaced with 3s next
cesd$b39 <- ifelse(cesd$b39 == 1, 3, cesd$b39)
cesd$b39 <- ifelse(cesd$b39 == 100, 1, cesd$b39) # Now swap placeholder
cesd$b39 <- ifelse(cesd$b39 == 4, 0, cesd$b39)

# 4. Sum the scores and binarize s.t. score >= 10 -> 1, else -> 0
score <- rowSums(cesd %>% select(-id))
score <- ifelse(score >= cesd_cutoff, 1, 0)

# Create DF with score and ID
cesd_df <- data.frame(score, id = cesd$id)

# Get all the other predictors to make a minimal model
df <- df %>% select(id,
                    residenc,   # Urban vs rural
                    a1,         # Sex
                    trueage,    # Age
                    b11,        # Satisfaction with life
                    b12,        # Satisfaction with health
                    f64a,       # Do you receive a retirement pension at the moment?
                    f1,         # Years attended school
                    f221,       # What is your monthly pension?
                    f35,        # Income per capita in the household
                    b310b,      # Hours of sleep per night
                    b121,       # Health compared to 1 year ago
                    g106,       # Difficulties hearing
                    f41,        # Current marital status
                    d91         # Exercise or not
                    )        

# Match the IDs to those with full CESD data
df <- inner_join(df, cesd_df, by = "id")

# Exclude anyone under the age of 65
df <- df %>% filter(trueage >= age_cutoff | is.na(trueage))

# Recode sex to match CHARLS: 0 = female, 1 = male
df$a1 <- ifelse(df$a1 == 2, 0, df$a1)

# Cleaning up satisfaction
df$b11 <- ifelse(df$b11 == 8, NA, df$b11)
df$b12 <- ifelse(df$b12 == 8, NA, df$b12)

# Cleaning up and coding income + pension for total income
df$f221 <- ifelse(df$f221 >= 88888, NA, df$f221)
df$f35 <- ifelse(df$f35 == 88888, NA, df$f35)
df$f35 <- ifelse(df$f35 == 99999, NA, df$f35)

pension_yesno <- ifelse(is.na(df$f64a) == TRUE, 0.9999, df$f64a) # Placeholder so if statements (below) evaluate
pension <- c()

for(i in 1:length(pension_yesno)){
  
  if(pension_yesno[i] == 0){
    
    pension[i] = 0
  }else(
    
    pension[i] = df$f221[i]
  )
}
pension_yearly <- pension*12

income_total <- pension_yearly + df$f35

# Add income, remove root variables
df <- df %>% select(-f221, -f35, -f64a) %>% mutate(income_total = income_total)

# Coding education
df$f1 <- ifelse(df$f1 == 88, NA, df$f1)
df$f1 <- ifelse(df$f1 == 99, NA, df$f1)

df$f1 <- ifelse(df$f1 == 0, 1999, df$f1) # No formal education
df$f1 <- ifelse(df$f1 < 6, 2999, df$f1) # Did not finish primary school
df$f1 <- ifelse(df$f1 == 6, 4999, df$f1) # Elementary school
df$f1 <- ifelse(df$f1 <= 9, 5999, df$f1) # Middle school
df$f1 <- ifelse(df$f1 <= 12, 6999, df$f1) # High school
df$f1 <- ifelse(df$f1 <= 15, 8999, df$f1) # 2/3 year associates degree
df$f1 <- ifelse(df$f1 <= 16, 9999, df$f1) # bachelors degree
df$f1 <- ifelse(df$f1 <= 20, 10999, df$f1) # Master's degree
df$f1 <- ifelse(df$f1 <= 65, 11999, df$f1) # PhD

# Convert to same scale as CHARLS
df$f1 <- (df$f1 - 999)/1000

# Convert missing hours of sleep (99) to actual NA values
df$b310b <- ifelse(df$b310b == 99, NA, df$b310b)

# Map health compared to last year to CHARLS values
df$b121 <- ifelse(df$b121 == 8, NA, df$b121)
df$b121 <- ifelse(df$b121 == 2, 1, df$b121) # Collapse into one "better" category
df$b121 <- ifelse(df$b121 == 3, 2, df$b121) # Shift "same" category down one
df$b121 <- ifelse(df$b121 == 4, 3, df$b121) # Collapse into one "worse" category and shift down 
df$b121 <- ifelse(df$b121 == 5, 3, df$b121)

# Map difficulties hearing to CHARLS values (1 = difficulties, 2 = no)
df$g106 <- ifelse(df$g106 == 1, 5, df$g106)

# Map marital status to CHARLS values
df$f41 <- ifelse(df$f41 == 8, NA, df$f41)
df$f41 <- ifelse(df$f41 == 9, NA, df$f41)
df$f41 <- ifelse(df$f41 == 5, 6, df$f41)
df$f41 <- ifelse(df$f41 == 4, 5, df$f41)
df$f41 <- ifelse(df$f41 == 3, 4, df$f41)

# Recode exercise to match CHARLS values
df$d91 <- ifelse(df$d91 == 8, NA, df$d91)
df$d91 <- ifelse(df$d91 == 2, 0, df$d91)

# Rename everything to match the names in CHARLS
df <- df %>% rename(age = trueage) %>%
             rename(ba000_w2_3 = a1) %>% 
             rename(dc028 = b11) %>%
             rename(da002 = b12) %>%
             rename(bd001_w2_4 = f1) %>%
             rename(da049 = b310b) %>%
             rename(da002_w2_1 = b121) %>%
             rename(be001 = f41) %>%
             rename(da051_1_ = d91) %>%
             rename(da039 = g106) %>%
             rename(bb001_w3_2 = residenc)

# Save
write.csv(df, "Data/CLHLS.csv", row.names = FALSE)