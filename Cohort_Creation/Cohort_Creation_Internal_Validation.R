# Libraries
library(tidyverse)
library(haven)
library(caret)

seed <- 19970507
set.seed(seed)

# Reading in the data, must be in a file called "Data" in the working directory
demographics_df_2018 <- read_dta("Data/Demographic_Background.dta")
cognition_df_2018 <- read_dta("Data/Cognition.dta")
ind_income_df_2018 <- read_dta("Data/Individual_Income.dta")
health_status_df_2018 <- read_dta("Data/Health_Status_and_Functioning.dta")

# Setting values
cesd_cutoff <- 10 # CES-D scores greater than or equal to this value will be classified as depressed
age_cutoff <- 65  # For testing CGD-Risk on different age groups using 2018 CHARLS, change this

# Get IDs and CES-D questions
cesd <- cognition_df_2018 %>% select(ID, 
                                     dc009, 
                                     dc010, 
                                     dc011, 
                                     dc012, 
                                     dc013, 
                                     dc014, 
                                     dc015, 
                                     dc016, 
                                     dc017, 
                                     dc018) %>%
                              rename(question_1  = dc009) %>%
                              rename(question_2  = dc010) %>%
                              rename(question_3  = dc011) %>%
                              rename(question_4  = dc012) %>%
                              rename(question_5  = dc013) %>%
                              rename(question_6  = dc014) %>%
                              rename(question_7  = dc015) %>%
                              rename(question_8  = dc016) %>%
                              rename(question_9  = dc017) %>%
                              rename(question_10 = dc018) 

nrow(cesd) # n = 19744

# Exclude everyone who refused (9) or did not know (8) for CES-D questions
cesd <- cesd %>% filter(question_1  < 8) %>%
                 filter(question_2  < 8) %>%
                 filter(question_3  < 8) %>%
                 filter(question_4  < 8) %>%
                 filter(question_5  < 8) %>%
                 filter(question_6  < 8) %>%
                 filter(question_7  < 8) %>%
                 filter(question_8  < 8) %>%
                 filter(question_9  < 8) %>%
                 filter(question_10 < 8) 

nrow(cesd) # n = 15961

# Edit CES-D entries to match scores as described here: https://www.brandeis.edu/roybal/docs/CESD-10_website_PDF.pdf
cesd$question_1  <- cesd$question_1  - 1
cesd$question_2  <- cesd$question_2  - 1
cesd$question_3  <- cesd$question_3  - 1
cesd$question_4  <- cesd$question_4  - 1
cesd$question_6  <- cesd$question_6  - 1
cesd$question_7  <- cesd$question_7  - 1
cesd$question_9  <- cesd$question_9  - 1
cesd$question_10 <- cesd$question_10 - 1

cesd$question_5 <- ifelse(cesd$question_5 == 3, 100, cesd$question_5) # Placeholder since 1s get replaced with 3s next
cesd$question_5 <- ifelse(cesd$question_5 == 1, 3, cesd$question_5)
cesd$question_5 <- ifelse(cesd$question_5 == 100, 1, cesd$question_5) # Now swap placeholder
cesd$question_5 <- ifelse(cesd$question_5 == 4, 0, cesd$question_5)

cesd$question_8 <- ifelse(cesd$question_8 == 3, 100, cesd$question_8) # Placeholder since 1s get replaced with 3s next
cesd$question_8 <- ifelse(cesd$question_8 == 1, 3, cesd$question_8)
cesd$question_8 <- ifelse(cesd$question_8 == 100, 1, cesd$question_8) # Now swap placeholder
cesd$question_8 <- ifelse(cesd$question_8 == 4, 0, cesd$question_8)

# Sum scores for the outcome, cut at threshold defined at the beginning of this script
score <- rowSums(cesd %>% select(-ID))
score <- ifelse(score >= cesd_cutoff, 1, 0)

# Save CES-D df with IDs
cesd_df <- data.frame(score, ID = cesd$ID)

# Filter demographics DF by variables of interest
demographics_df_2018 <- demographics_df_2018 %>% select(ID,             # Person level identifier
                                                        ba000_w2_3,     # Sex
                                                        ba004_w3_1,     # Year of Birth on ID card
                                                        ba005_w4,       # DOB on ID card same as actual DOB
                                                        ba002_1,        # Actual year of birth
                                                        bd001_w2_4,     # Education
                                                        bb000_w3_1,     # Type of residential address
                                                        bb001_w3_2,     # Location of residential address
                                                        be001           # Marital Status 
)     


# Filter health status DF by variables of interest 
health_status_df_2018 <- health_status_df_2018 %>% select(ID,         # Person level ID
                                                          da002,      # Self reported health status
                                                          da002_w2_1, # Health status compared to last survey
                                                          da049,      # Average hours of sleep/night
                                                          db001,      # Difficulty running or jogging 1km
                                                          db002,      # Difficulty walking 1km
                                                          db003,      # Difficulty walking 100m
                                                          db004,      # Difficulty getting up from chair
                                                          db005,      # Difficulty climbing multiple flights of stairs w/no rest
                                                          db006,      # Difficulty kneeling/crouching
                                                          db007,      # Difficulty extending arms
                                                          db008,      # Difficulty carrying weights >10 jin
                                                          db009,      # Difficulty picking up small coin
                                                          da034,      # Self reported eyesight up close 
                                                          da039,      # Self reported hearing status 
                                                          da041_w4,   # Trouble with body pain 
                                                          da051_1_    # Intense physical activity >10 mins
)      

# Filtering household income DF by variables of interest
ind_income_df_2018 <- ind_income_df_2018 %>% select(ID,               # Person level ID
                                                    ga001,            # Received any income Y/N
                                                    ga002,            # How much income 
                                                    ga003_w4_s1,      # Received pension Y/N
                                                    ga003_w4_1)       # Pension amount 

# Getting satisfaction variables
cognition_df_2018 <- cognition_df_2018 %>% select(ID,
                                                  dc028,              # Satisfaction with life
                                                  dc042_w3,           # Satisfaction with health
                                                  )           

# Join data frames by ID (only IDs with complete CES-D responses)
df <- inner_join(cesd_df, demographics_df_2018, by = "ID")
df <- inner_join(df, health_status_df_2018, by = "ID")
df <- inner_join(df, ind_income_df_2018, by = "ID")
df <- inner_join(df, cognition_df_2018, by = "ID")

nrow(df) # n = 15944

# ***************************************************************************************************************************************************
# Coding Age
# ***************************************************************************************************************************************************

ba005_w4 <- df$ba005_w4     # DOB on ID card same as actual DOB
ba004_w3_1 <- df$ba004_w3_1 # Year of Birth on ID card
ba002_1 <- df$ba002_1       # Actual year of birth if ID is wrong

ba005_w4 <- ifelse(is.na(ba005_w4) == TRUE, 9999, ba005_w4) # Placeholder so if statements (below) evaluate

birth_year <- c()

for(i in 1:nrow(df)){
  
  if(ba005_w4[i] == 1){ 
    
    birth_year[i] = ba004_w3_1[i]
    
  }else if(ba005_w4[i] == 2){
    
    birth_year[i] = ba002_1[i]
    
  }else if(ba005_w4[i] == 9999)
    
    birth_year[i] = 9999
}

# Calculate Age
age <- 2018 - birth_year

# Replace any NA placeholder values back with NAs
na_val <- 2018 - 9999
age <- ifelse(age == na_val, NA, age)

# Attach age to the data set and remove all root variables
df$age <- age
df <- df %>% select(-ba005_w4, -ba004_w3_1, -ba002_1)

# Filter by age >= 65 for elderly, keep NA values
df <- df %>% filter(age >= age_cutoff | is.na(age))

nrow(df) # n = 5681

# ***************************************************************************************************************************************************
# Coding Income 
# ***************************************************************************************************************************************************

ga001 <- df$ga001 # Received income Yes/No
ga002 <- df$ga002 # How much income

ga002 <- ifelse(is.na(ga002) == TRUE, 0.9999, ga002) # Placeholder so if statements (below) evaluate

income <- c()

for(i in 1:nrow(df)){
  
  if(ga001[i] == 2){
    
    income[i] = 0 # if answered "No" to income, then income amount is 0
    
  }else if(ga001[i] == 1){
    
    income[i] = ga002[i] # Otherwise income amount is reported income amount
    
  }
}

# Put NAs back
income <- ifelse(income == 0.9999, NA, income)

# Attach income column to df and remove all income related columns
df$income <- income
df <- df %>% select(-ga001, -ga002) 

# ***************************************************************************************************************************************************
# Coding Pension Amount
# ***************************************************************************************************************************************************

ga003_w4_s1 <- df$ga003_w4_s1 # Received any pension Yes/No
ga003_w4_1  <- df$ga003_w4_1  # How much pension

ga003_w4_s1 <- ifelse(is.na(ga003_w4_s1) == TRUE, 0.9999, ga003_w4_s1) # Placeholder so if statements (below) evaluate
ga003_w4_1 <- ifelse(is.na(ga003_w4_1) == TRUE, 0.9999, ga003_w4_1)    # Placeholder so if statements (below) evaluate

pension <- c()

for(i in 1:nrow(df)){
  
  if(ga003_w4_s1[i] == 0){
    
    pension[i] = 0 # If answered "No" to pension, then pension amount is 0
    
  } else if(ga003_w4_s1[i] == 1){
    
    pension[i] = ga003_w4_1[i]  # Otherwise pension amount is reported pension amount
  }
}

# Put NAs back
pension <- ifelse(pension == 0.9999, NA, pension)

# Attach pension to df and remove all pension related columns
df$pension <- pension
df <- df %>% select(-ga003_w4_1, -ga003_w4_s1) 

# ***************************************************************************************************************************************************
# Coding Difficulty Walking 100m
# ***************************************************************************************************************************************************
db001 <- df$db001
db002 <- df$db002

db001 <- ifelse(is.na(db001) == TRUE, 999, db001) # Placeholder so if statements evaluate
db002 <- ifelse(is.na(db002) == TRUE, 999, db002) # Placeholder so if statements evaluate

db003 <- df$db003

diff_walking_km <- c()
diff_walking    <- c()

# First start from the jog 1km and re-code for the walk 1km
for(i in 1:nrow(df)){
  
  if(db001[i] == 1){
    
    diff_walking_km[i] = 1 # If no difficulty jogging 1km, no difficulty walking 1km
    
  }else if(db001[i] == 2 || db001[i] == 3 || db001[i] == 4){
    
    diff_walking_km[i] = db002[i] # Otherwise use reported difficulty
    
  }else if(db001[i] == 999){
    
    diff_walking_km[i] = 999
  }
}

# Now re-code difficulty walking 100m 
for(i in 1:nrow(df)){
  
  if(diff_walking_km[i] == 1){
    
    diff_walking[i] = 1 # If no difficulty walking 1km, no difficulty walking 100m
    
  }else if(diff_walking_km[i] == 2 || diff_walking_km[i] == 3 || diff_walking_km[i] == 4){
    
    diff_walking[i] = db003[i] # Otherwise keep entry for 100m difficulty
  }
}

# Put NAs back
diff_walking_km <- ifelse(diff_walking_km == 999, NA, diff_walking_km)
diff_walking <- ifelse(diff_walking == 999, NA, diff_walking)

# Add to df and remove root variables
df$diff_walking <- diff_walking
df$diff_walking_km <- diff_walking_km
df <- df %>% select(-db003, -db002, -db001) 

# Recode any Yes/No questions as binary instead of 1/2, w/ 1 = YES and 0 = NO
df$ba000_w2_3 <- ifelse(df$ba000_w2_3 == 2, 0, df$ba000_w2_3)
df$da051_1_   <- ifelse(df$da051_1_ == 2, 0, df$da051_1_)

# Code questions that had "Don't Know' responses as missing
df$da034 <- ifelse(df$da034 == 997, NA, df$da034)
df$da039 <- ifelse(df$da039 == 997, NA, df$da039)

# Adding total income and removing root variables
df <- df %>% mutate(income_total = income + pension) %>% select(-income, -pension)

# n for 2018 data 
nrow(df) # n = 5681

# Splitting into training and testing sets: 80% train 20% test
train_ind <- createDataPartition(y = df$score, times = 1, p = 0.8)

df_train <- df[train_ind[[1]],]
df_test <- df[-train_ind[[1]],]

# Saving all of the data sets
write.csv(df, "Data/CHARLS_complete.csv", row.names = FALSE)
write.csv(df_train, "Data/CHARLS_train.csv", row.names = FALSE)
write.csv(df_test, "Data/CHARLS_test.csv", row.names = FALSE)