####################### Script Comments #######################
#**************************************************************
#*
#*  XGBoost - Classification - Adult Dataset
#*
#*  Objective: Explore some of the packages and techniques
#*    associated with XGBoost. Import Adult education dataset
#*    from UCI Machine Learning Respository to work with.
#*
#*  Initial Build - 7/4/2018 - Chris Castillo
#*
#*  Change Log
#*  m/d/yyyy
#*    - 
#*
#*  Notes:
#*    - Aggregating all R scripts into a singular script for 
#*      Github upload
#*    - Initial XGBoost Results: Train AUC = 0.953397, Test CV  
#*      AUC = 0.929003, Validate AUC = 0.9277
#*    - MLR Random Parameter XGBoost Results: Train AUC =  
#*      0.958285, Test CV AUC = 0.9286709, Validate AUC = 0.9286
#*    - MLR Grid Search XGBoost Results: Train AUC = 0.978390,
#*      Test CV AUC = 0.9291430, Validate AUC = 0.9222
#**************************************************************
#**************************************************************


#* Clear workspace 
rm(list=ls())
gc()


#* Load libraries
library(data.table)
library(ggplot2)
library(plyr)
library(dplyr)
library(h2o)
library(xgboost)
library(parallel)
library(parallelMap)
library(mlr)
library(caret)
library(doParallel)
library(pROC)


#* Define Model.Target to be what we're solving for
Model.Target <- "income"


#* Define Random.Seed for reproducibility
Random.Seed <- 10


#* Establish random seed for reproducibility
set.seed(Random.Seed) # Need to figure how to get reproducible results, while parallelizing mlr tuneParams, currently not possible for Windows


#**********************************************
#**********************************************
##### Import Training Dataset And Cleanse #####
#**********************************************
#**********************************************

#* Import Train dataset from UCI Machine Learning Repository
Data <- fread(
  input = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  , stringsAsFactors = TRUE
  , strip.white = TRUE
  , data.table = FALSE
  , header = FALSE
)



#* Append on variable names | should eventually scrape directly from website
colnames(Data) <- c(
  "age"
  , "workclass"
  , "fnlwgt"
  , "education"
  , "education-num"
  , "marital-status"
  , "occupation"
  , "relationship"
  , "race"
  , "sex"
  , "capital-gain"
  , "capital-loss"
  , "hours-per-week"
  , "native-country"
  , "income"
)



#* Fix column names with "-" in the name
names(Data) <- gsub(pattern = "-"
                    , replacement = "_"
                    , names(Data)
)



#* Find the list of data frame names and their respective classes 
allClass <- function(x)
{unlist(lapply(unclass(x), class))}



#* Create a list of the columns that are class = integer 
allClass(Data)
Factor.List <- names(Data)[allClass(Data) == "factor"]
Integer.List <- names(Data)[allClass(Data) %in% c("integer", "numeric")]



#* Replace "" records with "_Blank_" & "NA" or "?" as "_Unknown_"
for (i in 1:length(Factor.List)){
  
  Data[ , Factor.List[i]] <- as.character(Data[ , Factor.List[i]])
  
  
  Data[ , Factor.List[i]][ Data[ ,Factor.List[i]] == "" ] <- "_Blank_"
  
  Data[ , Factor.List[i]][is.na(Data[ ,Factor.List[i]])] <- "_Unknown_"
  
  Data[ , Factor.List[i]][ Data[ ,Factor.List[i]] == "?" ] <- "_Unknown_"
  
  Data[ , Factor.List[i]] <-gsub(pattern = "-"
                                 , replacement = "_"
                                 , x = Data[ , Factor.List[i]]
  )
  
  
  Data[ , Factor.List[i]] <- as.factor(Data[ , Factor.List[i]])
  
}


#* Create a data.frame to populate with Data factors and the level counts
Data_Levels_Table <- data.frame(
  matrix(
    ncol = 2
    , nrow = length(Factor.List)
  )
)
colnames(Data_Levels_Table) <- c(
  "Factor"
  , "Level_Count"
)


#* Populate table with factors and level counts
for (i in 1:length(Factor.List)){
  
  Data_Levels_Table[ i, 1 ] <- Factor.List[i]
  Data_Levels_Table[ i , 2 ] <- length(
    levels(
      eval(
        parse(
          text = paste(
            "Data$"
            , Factor.List[i]
            , sep = ""
          )
        )
      )
    )
  )
  
}


#* Create Train.Data from Data
Train.Data <- Data
rm(Data)


#******************************************************************
#******************************************************************
##### Store levels of Train.Data factors for level correction #####
#******************************************************************
#******************************************************************


#* Find the list of data frame names and their respective classes 
allClass <- function(x)
{unlist(lapply(unclass(x), class))}


#* Create a list of the columns that are class = factor
allClass(Train.Data)
Factor.List.Train <- names(Train.Data)[allClass(Train.Data) == "factor"]


#* Exclude Model.Target from Factor.List.Train if it exists
if(exists("Model.Target")){
  
  Factor.List.Train <- Factor.List.Train[ !Factor.List.Train %in% Model.Target ]
  
}


#* Create name vectors for each factor that stores the different levels in Train.Data
for (i in 1:length(Factor.List.Train)){
  
  assign(paste("Levels.Train.", Factor.List.Train[i], sep = ""), levels(eval(parse(text = paste("Train.Data$", Factor.List.Train[i], sep="")))))
  
}


#* Create a list of factor vectors
Train.Level.Objects <- ls()[grep('^Levels.Train.*?', ls())]


#* Create a vector that will store factors that contain _Unknown_
Unknown_Match <- NULL
#* If one of the factors contains _Unknown_ as a level, store the factor in Unknown_Match
for (i in 1:length(Factor.List.Train))
{if(is.na(match('_Unknown_', get(ls()[grep('^Levels.Train.*?', ls())][i]))) == FALSE)
{Unknown_Match <- c(Unknown_Match, ls()[grep('^Levels.Train.*?', ls())][i])}
  
  # print(paste(ls()[grep('^Levels.Train.*?', ls())][i],
  #             if(is.na(match('_Unknown_', get(ls()[grep('^Levels.Train.*?', ls())][i]))) == FALSE)
  #             {'Match'} else {'No Match'}
  #             , sep = " "
  # )
  # )
} #* Close loop


#* Store all factors that don't have _Unknown_
Missing_Unknown_Factors <- Train.Level.Objects[!Train.Level.Objects %in% Unknown_Match]


#* Create a vector that will store factors that contain _Other_
Other_Match <- NULL
#* If one of the factors contains _Other_ as a level, store the factor in Other_Match
for (i in 1:length(Factor.List.Train))
{if(is.na(match('_Other_', get(ls()[grep('^Levels.Train.*?', ls())][i]))) == FALSE)
{Other_Match <- c(Other_Match, ls()[grep('^Levels.Train.*?', ls())][i])}
  
  # print(paste(ls()[grep('^Levels.Train.*?', ls())][i],
  #             if(is.na(match('_Other_', get(ls()[grep('^Levels.Train.*?', ls())][i]))) == FALSE)
  #             {'Match'} else {'No Match'}
  #             , sep = " "
  # )
  # )
} #* Close loop


#* Store all factors that don't have _Other_
Missing_Other_Factors <- Train.Level.Objects[!Train.Level.Objects %in% Other_Match]


#******************************************
#******************************************
##### Import Test Dataset And Cleanse #####
#******************************************
#******************************************

#* Import Test dataset from UCI Machine Learning Repository
Score.Import <- fread(
  input = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
  , stringsAsFactors = TRUE
  , strip.white = TRUE
  , data.table = FALSE
  , header = FALSE
)


#* Append on variable names | should eventually scrape directly from website
colnames(Score.Import) <- c(
  "age"
  , "workclass"
  , "fnlwgt"
  , "education"
  , "education-num"
  , "marital-status"
  , "occupation"
  , "relationship"
  , "race"
  , "sex"
  , "capital-gain"
  , "capital-loss"
  , "hours-per-week"
  , "native-country"
  , "income"
)


#* Fix column names with "-" in the name
names(Score.Import) <- gsub(pattern = "-"
                            , replacement = "_"
                            , names(Score.Import)
)


#***********************************
#***********************************
##### Score Dataset Imputation #####
#***********************************
#***********************************

#* Find the list of data frame names and their respective classes 
allClass <- function(x)
{unlist(lapply(unclass(x), class))}


#* Create a list of the columns that are class = factor and class = integer 
allClass(Score.Import)
Factor.List.Score <- names(Score.Import)[allClass(Score.Import) == "factor"]
Integer.List.Score <- names(Score.Import)[allClass(Score.Import) %in% c("integer", "numeric")]


#* Exclude Model.Target from Factor.List.Score if it exists
if(exists("Model.Target")){
  
  Factor.List.Score <- Factor.List.Score[ !Factor.List.Score %in% Model.Target ]
  
}


#* Replace "" records with "_Blank_" & "NA" as "_Unknown_"
for (i in 1:length(Factor.List.Score)){
  
  Score.Import[ , Factor.List.Score[i]] <- as.character(Score.Import[ , Factor.List.Score[i]])
  
  Score.Import[ , Factor.List.Score[i]][ Score.Import[ , Factor.List.Score[i]] == "" ] <- "_Blank_"
  
  Score.Import[ , Factor.List.Score[i]][is.na(Score.Import[ ,Factor.List.Score[i]])] <- "_Unknown_"
  
  Score.Import[ , Factor.List.Score[i]][ Score.Import[ ,Factor.List.Score[i]] == "?" ] <- "_Unknown_"
  
  Score.Import[ , Factor.List.Score[i]] <-gsub(pattern = "-"
                                               , replacement = "_"
                                               , x = Score.Import[ , Factor.List.Score[i]]
  )
  
  
  Score.Import[ , Factor.List.Score[i]] <- as.factor(Score.Import[ , Factor.List.Score[i]])
  
}


#* Drop and unnecessary levels
Score.Import <- droplevels(Score.Import)


#* Identify any factors that still have "" levels
for (i in 1:length(Factor.List.Score)) {
  
  z <- levels(Score.Import[ , Factor.List.Score[i]])
  
  if (length(z[z == ""]) != 0){
    
    print(
      paste(
        Factor.List.Score[i]
        , "contains Blanks"
        , sep = " "
      )
    )
    
  }
  
  y <- length(Score.Import[ , Factor.List.Score[i]][is.na(Score.Import[ , Factor.List.Score[i]])])
  
  if (y != 0){
    
    print(
      paste(
        Factor.List.Score[i]
        , "contains NAs"
        , sep = " "
      )
    )
    
  }
  
  #* Clear out temporary objects
  rm(z, y)
  
} ## Close main loop


#**********************************
#**********************************
##### Level Vector Comparison #####
#**********************************
#**********************************

#* Ensure that Score.Data removes variables that are in the Exclude list
if(exists("Exclude")){
  
  if(length(Exclude) != 0){
    
    Score.Data <- Score.Import[ , -which(names(Score.Import) %in% Exclude)]
    
  }
  
} else {
  
  Score.Data <- Score.Import[ , ]
  
}


#* Remove any superfluous factor levels
Score.Data <- droplevels(Score.Data)


#* Check to see if the factor names are present in Train.Data and Score.Data
if(!identical(Factor.List.Train, Factor.List.Score)){
  break()
}


#* Create name vectors for each factor that stores the different levels in Score.Data
for (i in 1:length(Factor.List.Score)){
  
  assign(paste("Levels.Score.", Factor.List.Score[i], sep = ""), levels(eval(parse(text = paste("Score.Data$", Factor.List.Score[i], sep = "")))))
  
} #* Close loop


#* Loop through level comparisons and paste differences, store factors that need to have missing levels coerced
Missing_Level_Factors <- NULL
for (i in 1:length(Factor.List.Score)){
  
  Score.Compare <- get(ls()[
    which(
      ls() == paste("Levels.Score."
                    , Factor.List.Score[i]
                    , sep = ""
      )
    )
    ]
  )
  
  Train.Compare <- get(ls()[
    which(
      ls() == paste("Levels.Train."
                    , Factor.List.Score[i]
                    , sep = ""
      )
    )
    ]
  )
  
  
  Missing <- Score.Compare[ !Score.Compare %in% Train.Compare]
  
  
  if (length(Missing) != 0){
    
    for (j in 1:length(Missing)){
      
      print(paste(Missing[j]
                  , " is a level missing from Levels.Train."
                  , Factor.List.Score[i]
                  , sep = ""
      )
      )
      
      Missing_Level_Factors <- c(Missing_Level_Factors
                                 , Factor.List.Score[i]
      )
      
    } #* Close print/paste LOOP
  } #* Close Missing != 0 check IF
  
  rm(Score.Compare, Train.Compare)
  
} #* Close final LOOP that iterates through the number of stored factor levels
Missing_Level_Factors <- unique(Missing_Level_Factors)


#* Print the factors that need to have missing levels imputed
print("The following categorical variables are missing a level compared to Train.Data")
print(Missing_Level_Factors)


#*****************************************
#*****************************************
##### Level cleansing for Score Data #####
#*****************************************
#*****************************************


#* Append on any factor levels in the Train.Data that is not in the Score.Data
for (i in 1:length(Factor.List.Score)){
  
  Train.Data.Level <- eval(parse(text = paste("Train.Data$", Factor.List.Score[i], sep = "")))
  Score.Data.Level <- eval(parse(text = paste("Score.Data$", Factor.List.Score[i], sep = "")))
  
  levels(Score.Data[ , match(Factor.List.Score[i], names(Score.Data))]) <- c(levels(Score.Data.Level), levels(Train.Data.Level)[!levels(Train.Data.Level) %in% levels(Score.Data.Level)])
  #print(paste("Train.Data$", Factor.List.Score[i]," vs. ", "Score.Data$", Factor.List.Score[i], sep = ""))
  
} #* Close loop


#* Find the object names of any other categorical factors that need to have missing levels coerced
Missing_Level_Factors <- Missing_Level_Factors[ !Missing_Level_Factors %in% unique(gsub("Levels.Train."
                                                                                        , ""
                                                                                        , c(Other_Match
                                                                                            #, Unknown_Match # Removed the need for coercing to _Unknown_
                                                                                        )
))
]


#* Coerce any Score.Data factor level that doesn't exist in Train.Data to "_Other_"
if(length(Other_Match) > 0){
  for (i in 1:length(Other_Match)){
    
    StoreVar <- match(substr(Other_Match[i], nchar("Levels.Train.") + 1, nchar(Other_Match)[i]), names(Score.Data))
    Score.Data[ , StoreVar][!(Score.Data[ , StoreVar]) %in% get(paste(Other_Match[i], sep = ""))] <- "_Other_"
    
  } #* Close loop
} #* Close loop


#* Coerce any Score.Data factor level that doesn't exist in Train.Data to "_Unknown_"
#if(length(Unknown_Match) > 0){
#for (i in 1:length(Unknown_Match)){

#StoreVar <- match(substr(Unknown_Match[i], nchar("Levels.Train.") + 1, nchar(Unknown_Match)[i]), names(Score.Data))
#Score.Data[ , StoreVar][!(Score.Data[ , StoreVar]) %in% get(paste(Unknown_Match[i], sep = ""))] <- "_Unknown_"

#} #* Close loop
#}


#* Coerce other missing factor levels with "Smart Level Imputation"
if(length(Missing_Level_Factors) > 0){
  for(i in 1:length(Missing_Level_Factors)){
    
    StoreVar <- Missing_Level_Factors[i]
    
    Missing_Levels <- get(paste("Levels.Score.", StoreVar, sep = ""))[!get(paste("Levels.Score.", StoreVar, sep = "")) %in% get(paste("Levels.Train.", StoreVar, sep = ""))]      
    
    Missing_Count <- length(eval(parse(text = paste("Score.Data$", StoreVar, sep = "")))[ eval(parse(text = paste("Score.Data$", StoreVar, sep = ""))) %in% Missing_Levels ])
    if(Missing_Count > 0){
      print(paste(StoreVar, " IS being imputed", sep = ""))
    } #else {
    #print(paste(StoreVar, " is NOT missing a level and will NOT be imputed", sep = ""))
    #}
    
    #* Set seed to ensure reproducibility for imputation
    set.seed(10)
    Score.Data[ , StoreVar][ Score.Data[ , StoreVar] %in% Missing_Levels] <- sample(eval(parse(text = paste("Train.Data$", StoreVar, sep = ""))), Missing_Count, replace = TRUE)
    
  }
} ## Close loop


#* Clear out any NULL factor levels
Score.Data <- droplevels(Score.Data)


#* Re-order Score.Data factor levels to align with Train.Data factor levels
for (i in 1:length(Factor.List.Score)){
  
  Score.Data[ , match(Factor.List.Score[i], names(Score.Data))] <- factor(Score.Data[ , match(Factor.List.Score[i], names(Score.Data))], levels = get(paste("Levels.Train.", Factor.List.Score[i], sep = "")))
  
} #* Close loop


#******************************************
#******************************************
##### Impute Missing Numerical Values #####
#******************************************
#******************************************

#* Use the median value of integer fields within Train.Data to impute NA values within Score.Data
for (i in 1:length(Integer.List.Score)){
  
  Score.Data[ 
    , match(Integer.List.Score[i]
            , names(Score.Data)
    )
    ][
      is.na(
        Score.Data[ 
          , match(Integer.List.Score[i]
                  , names(Score.Data)
          )
          ]
      )
      ] <- median(
        x = Train.Data[ 
          , match(Integer.List.Score[i]
                  , names(Train.Data)
          )
          ]
        , na.rm = TRUE
      )
  
} #* Close loop


#* Ensure Score.Data hasn't lost any records from the data cleansing
if(!identical(nrow(Score.Data), nrow(Score.Data[ complete.cases(Score.Data), ]))){
  
  break()
  
}


#* Remove any NA datum records
Score.Data <- Score.Data[ complete.cases(Score.Data), ]


#* Memory cleanup
gc()


#************************************************************************************
#************************************************************************************
##### Transform Train Data for XGBoost & Train Model Using Arbitrary Parameters #####
#************************************************************************************
#************************************************************************************

#* Create model.matrix object from Train.Data
Train.x <- model.matrix( ~ . + 0
                         , data = Train.Data[ , !(names(Train.Data) %in% Model.Target)]
                         #, with = FALSE
)


#* Create vector of Model.Target, reframe target variables into (0, 1) outcome, reclassify as.numeric
Train.y <- Train.Data[ , Model.Target]
Train.y <- as.numeric(Train.y) - 1


#* Use XGBoost matrix conversion function
dTrain <- xgb.DMatrix(data = Train.x
                      , label = Train.y # "label" is our target variable
)


#* Define default.parameters for XGBoost training
default.parameters <- list(
  booster = "gbtree"
  , objective = "binary:logistic"
  , eta = 0.3 # default learning rate for gradient descent steps
  , gamma = 0 # default regularization (large non-performant coefficient penalization)
  , max_depth = 6 # default depth of decision tree
  , min_child_weight = 1 # default number of observations with a child node 
  , subsample = 1
  , colsample_bytree = 1
)


#* Store system time before model run
Start.Time <- Sys.time()

#* Train XGBoost model using 5-fold cross-validation, optimize for AUC
set.seed(Random.Seed) # Set for reproducibile results
Train.xgbcv <- xgb.cv(
  params = default.parameters
  , data = dTrain # Use 
  , nrounds = 100 # Number of iterations to try
  , nfold = 5 # 5-fold CV
  , showsd = TRUE # Show std dev of cross-validation metric result of each iteration
  , stratified = TRUE # Stratify the fold sampling by outcome values, target distribution is maintained
  , print_every_n = 1 # Display status every n number of iterations
  , early_stopping_rounds = 20 # If performance doesn't increase for 20 consecutive iterations, STOP
  , metrics = "auc" # Define evaluation metric
  , maximize = TRUE # The larger the evaluation metric the better, FALSE is the opposite
)

#* Print out model training runtime
print(Sys.time() - Start.Time)


#* Best Iteration [63]: Train AUC = 0.953397, Test AUC = 0.929003
Train.xgbcv


#* Create xgboost object from optimized parameters, just nrounds in this case
Train.xgb <- xgboost(
  data = dTrain
  , nrounds = Train.xgbcv$best_iteration
  , params = default.parameters
  , verbose = 1
)


#*********************************************
#*********************************************
##### Score Validation data (Score.Data) #####
#*********************************************
#*********************************************

#* Create model.matrix object from Score.Data
Score.x <- model.matrix( ~ . + 0
                         , data = Score.Data[ , !(names(Score.Data) %in% Model.Target)]
                         #, with = FALSE
)


#* Create vector of Model.Target, reframe target variables into (0, 1) outcome, reclassify as.numeric
Score.y <- Score.Data[ , Model.Target]
Score.y <- as.numeric(Score.y) - 1


#* Use XGBoost matrix conversion function
dScore <- xgb.DMatrix(data = Score.x
                      , label = Score.y # "label" is our target variable
)


#* Create prediciton set from Train.xgb on dScore (XGBoost matrix conversion)
Score.predict <- predict(
  object = Train.xgb
  , newdata = dScore
)


#* Determine Score ROC statistics (AUC = 0.9277)
roc(
  response = Score.y
  , predictor = Score.predict
  , auc = TRUE
  , plot = TRUE
)
  

#*****************************************************************************************************************
#*****************************************************************************************************************
##### Set Test.Data and run CV XGBoost parameter training using MLR with random constrained parameter search #####
#*****************************************************************************************************************
#*****************************************************************************************************************

#* Create tasks for Train.Data
Train.Task <- makeClassifTask(data = Train.Data
                              , target = Model.Target
                              , positive = ">50K"
)


#* Do OHE (One-Hot-Encoding) for defined tasks
Train.Task <- createDummyFeatures(obj = Train.Task
                                  #, target = "income" # Getting an error for trying to pass a target
)


#* Create Learner (mlr)
Train.lrn <- makeLearner("classif.xgboost"
                         , predict.type = "prob"
)


#* Fill Learner parameters
Train.lrn$par.vals <- list(
  objective = "binary:logistic"
  , eval_metric = "auc"
  , nrounds = 100
  , eta = 0.1
)


#* Set parameter space
Train.parameters <- makeParamSet(
  makeDiscreteParam("booster"
                    , values = "gbtree"
  )
  , makeIntegerParam("max_depth"
                     , lower = 3
                     , upper = 10
  )
  , makeNumericParam("min_child_weight"
                     , lower = 1
                     , upper = 10
  )
  , makeNumericParam("subsample"
                     , lower = 0.5
                     , upper = 1
  )
  , makeNumericParam("colsample_bytree"
                     , lower = 0.5
                     , upper = 1
  )
)


#* Establish resampling strategy
Train.rdesc <- makeResampleDesc("CV"
                                , stratify = TRUE
                                , iters = 5L
)


#* Search strategy, set to 100 iterations
Train.ctrl <- makeTuneControlRandom(maxit = 100L)


#* Establish parallel backend
parallelStart(mode = "socket"
              , cpus = detectCores() - 1
              , level = "mlr.tuneParams"
              #, mc.set.seed = TRUE # Only applies to UNIX OS, can't get repeatable results in Windows due to random 
              # resampling with parallel runs that I cannot pass the random.seed() to
)


#* Store system time for parameter tuning run
Start.Time <- Sys.time()

#* Start parameter tuning
Train.tune.random <- tuneParams(
  learner = Train.lrn
  , task = Train.Task
  , resampling = Train.rdesc
  , measures = mlr::auc
  , par.set = Train.parameters
  , control = Train.ctrl
  , show.info = TRUE
)

#* End parallelization
parallelStop()

#* Print time spend on parameter tuning
print(Sys.time() - Start.Time)


#* CV Test AUC = 0.9286709
print(Train.tune.random$y)


#* Set parameter values for learner equal to the optimized values 
Train.lrn <- setHyperPars(
  learner = Train.lrn
  , par.vals = Train.tune.random$x
)


#* Train up xgboost model from learner that now has optimized hyperparamaters
Train.xgb.lrn <- mlr::train(
  learner = Train.lrn
  , task = Train.Task
)


#* Create tasks for Score.Data
Score.Task <- makeClassifTask(data = Score.Data
                              , target = Model.Target
                              , positive = ">50K."
                              , fixup.data = "no" # Doesn't prevent empty levels from being dropped
                              , check.data = FALSE
)


#* Do OHE (One-Hot-Encoding) for defined tasks
Score.Task <- createDummyFeatures(obj = Score.Task
                                  #, target = "income" # Getting an error for trying to pass a target
)


#* Create a prediction set from Score.Task; will have to call $response
Score.predict <- predict(
  Train.xgb.lrn
  , Score.Task # dScore
)


#* Determine Score ROC statistics (AUC = 0.9286)
roc(
  response = Score.y
  , predictor = Score.predict$data[[4]]
  , auc = TRUE
  , plot = TRUE
)


#***************************************************************************************************
#***************************************************************************************************  
##### Set Test.Data and run CV XGBoost parameter training using MLR with parameter grid search #####
#***************************************************************************************************
#***************************************************************************************************

#* Create tasks for Train.Data and Test.Data
Train.Task <- makeClassifTask(data = Train.Data
                              , target = Model.Target
                              , positive = ">50K"
)


#* Do OHE (One-Hot-Encoding) for defined tasks
Train.Task <- createDummyFeatures(obj = Train.Task
                                  #, target = "income" # Getting an error for trying to pass a target
)


#* Create Learner (mlr)
Train.lrn <- makeLearner("classif.xgboost"
                         , predict.type = "prob"
)


#* Fill Learner parameters
Train.lrn$par.vals <- list(
  objective = "binary:logistic"
  , eval_metric = "auc"
  , nrounds = 100
)


#* Set parameter space
Train.parameters <- makeParamSet(
  makeDiscreteParam("booster"
                    , values = "gbtree"
  )
  , makeDiscreteParam("max_depth"
                      , values = c(2, 4, 6, 8, 10)
  )
  , makeDiscreteParam("min_child_weight"
                      , values = c(1, 2)
  )
  , makeDiscreteParam("subsample"
                      , values = 1
  )
  , makeDiscreteParam("colsample_bytree"
                      , values = 1
  )
  , makeDiscreteParam("eta"
                      , values = c(0.1, 0.2, 0.3)
  )
  , makeDiscreteParam("gamma"
                      , values = c(0, 5, 10, 20)
  )
)


#* Establish resampling strategy
Train.rdesc <- makeResampleDesc("CV"
                                , stratify = TRUE
                                , iters = 5L
)


#* Search strategy
Train.ctrl <- makeTuneControlGrid()


#* Establish parallel backend
parallelStart(mode = "socket"
              , cpus = detectCores() - 1
              , level = "mlr.tuneParams"
              #, mc.set.seed = TRUE # Only applies to UNIX OS, can't get repeatable results in Windows due to random 
              # resampling with parallel runs that I cannot pass the random.seed() to
)


#* Store system time for parameter tuning run
Start.Time <- Sys.time()

#* Start parameter tuning
Train.mlr.grid <- tuneParams(
  learner = Train.lrn
  , task = Train.Task
  , resampling = Train.rdesc
  , measures = mlr::auc
  , par.set = Train.parameters
  , control = Train.ctrl
  , show.info = TRUE
)

#* End parallelization
parallelStop()

#* Print time spend on parameter tuning, 33 minute runtime
print(Sys.time() - Start.Time)



#* Runtime: 33 minutes
#* Best model: max_depth = 6, eta = 0.2, gamma = 0, colsample_bytree = 1, min_child_weight = 1, subsample = 1
#* AUC.Test.Mean = 0.929125


#* Show Test AUC
print(Train.mlr.grid$y)


#* Set parameter values for learner equal to the optimized values 
Train.lrn <- setHyperPars(
  learner = Train.lrn
  , par.vals = Train.mlr.grid$x
)


#* Train up xgboost model from learner that now has optimized hyperparamaters
Train.xgb.lrn.grid <- mlr::train(
  learner = Train.lrn
  , task = Train.Task
)


#* Create tasks for Score.Data
Score.Task <- makeClassifTask(data = Score.Data
                              , target = Model.Target
                              , positive = ">50K."
                              , fixup.data = "no" # Doesn't prevent empty levels from being dropped
                              , check.data = FALSE
)


#* Do OHE (One-Hot-Encoding) for defined tasks
Score.Task <- createDummyFeatures(obj = Score.Task
                                  #, target = "income" # Getting an error for trying to pass a target
)


#* Create a prediction set from Score.Task; will have to call $response
Score.predict <- predict(
  Train.xgb.lrn.grid
  , Score.Task # dScore
)


#* Determine Score ROC statistics (AUC = 0.8942)
roc(
  response = Score.y
  , predictor = Score.predict$data[[4]]
  , auc = TRUE
  , plot = TRUE
)