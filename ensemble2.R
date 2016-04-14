# Initiate ----------------------------------------------------------------
source("common.R")

# Start Parallel Processing
cl <- makeCluster(max(detectCores()-1,1))
registerDoParallel(cl)


# Data --------------------------------------------------------------------
# Load tidy_train and tidy_test
load(file = file.path(".", "Data", "tidySets.Rda"))

set.seed(mySeed)

# Divide tidy_train into 3 sets:
trainIdx_1 <- createDataPartition(tidy_train$TARGET, p = 1/3, list = FALSE)

trainingSet_1  <- tidy_train[trainIdx_1, ]
trainingSet_23 <- tidy_train[-trainIdx_1, ]

trainIdx_2 <- createDataPartition(trainingSet_23$TARGET, p = 1/2, list = FALSE)

trainingSet_2 <- trainingSet_23[trainIdx_2, ]
trainingSet_3 <- trainingSet_23[-trainIdx_2, ]

trainingClass_1 <- factor(trainingSet_1$TARGET,
                          labels = c("SATISFIED", "UNSATISFIED"))
trainingClass_2 <- factor(trainingSet_2$TARGET,
                          labels = c("SATISFIED", "UNSATISFIED"))
trainingClass_3 <- factor(trainingSet_3$TARGET,
                          labels = c("SATISFIED", "UNSATISFIED"))

predLabels <- setdiff(names(tidy_train), c("ID", "TARGET"))

trainingSet_1 <- trainingSet_1[,predLabels]
trainingSet_2 <- trainingSet_2[,predLabels]
trainingSet_3 <- trainingSet_3[,predLabels]

save(trainingSet_1, trainingSet_2, trainingSet_3,
     trainingClass_1, trainingClass_2, trainingClass_3,
     file = file.path(".", "Data", "trainingSets_123.Rda"))

# Don't need this any more
rm (tidy_train)

# Build Single Models List ------------------------------------------------

# We'll build a bunch of single models training them on trainingSet_1.
# Later we'll validate them twofold: on CV basis and using trainingSet_2 as
# validation set
classifModels <- c("gbm",  "glm", "LogitBoost")
# classifModels <- c("gbm", "ranger", "svmRadial", "xgbLinear", "AdaBag",
#                    "AdaBoost.M1", "glm", "glmboost", "xgbTree", "LogitBoost")

trCtrl <- trainControl(method            = "cv",
                       number            = numCVs,
                       # repeats           = 3,
                       verboseIter       = TRUE,
                       savePredictions   = "final",
                       classProbs        = TRUE,
                       summaryFunction   = twoClassSummary,
                       allowParallel     = TRUE,
                       index             = createFolds(trainingClass_1,
                                                       numCVs))

tuneModels  <- list(nn  = caretModelSpec(method     = "nnet", 
                                         trace      = FALSE))

set.seed(mySeed)

modelsList <- caretList(# Arguments to pass to train() as '...' ###
  x          = trainingSet_1,
  y          = trainingClass_1, 
  metric     = "ROC",
  ###########################################
  # caretList() specific arguments ##########
  trControl  = trCtrl,
  methodList = classifModels,
  tuneList   = tuneModels
  ###########################################
)

save(modelsList,  file = file.path(".", "Models", "modelsList2.Rda"))


# Compare Models ----------------------------------------------------------

## ON CV basis
resamps <- resamples(modelsList)
summary(resamps)
modelsROC_CV <- summary(resamps)$statistics$ROC[,"Mean"]

xyplot(resamps, what = "BlandAltman")
dotplot(resamps)
densityplot(resamps)
bwplot(resamps)
splom(resamps)
parallelplot(resamps)

diffs <- diff(resamps)
summary(diffs)

## ON validation set trainingSet_2
modelsPerfVal <- modelsListPerformance(modelsList, 
                                       trainingSet_2, trainingClass_2, 
                                       lev = c("SATISFIED", "UNSATISFIED"))

modelsROC_Val <- modelsPerfVal[,"ROC"]

## CV and Validation Set Methods - summary
summaryModelsPerf <- modelsListPerf_CV_VAL(modelsList, 
                                           trainingSet_2, trainingClass_2, 
                                           lev = c("SATISFIED", "UNSATISFIED"))
summaryModelsPerf

# Choosing models ---------------------------------------------------------

## Best performers (based on CV)
numModels <- min(length(modelsList), maxBP)
bestModelsOrder <- sort(resamps, decreasing = TRUE, metric = "ROC")[1:numModels]
bestModelsOrder

best_modelsLists <- modelsList[names(modelsList[bestModelsOrder])]
class(best_modelsLists) <- "caretList"

## Low correlated models (based on ?? Could be both on training and on 
#  validation sets)
lowCorrModels <- getLowCorrModels(modelsList, trainingSet_2, "UNSATISFIED",
                                  cutoff = 0.75, printTable = TRUE)
lowCorrModels

LC_modelsLists <- modelsList[names(modelsList[lowCorrModels])]
class(LC_modelsLists) <- "caretList"

## From now on i don't need modelsList

# Ensemble 1: a caretList ensemble ----------------------------------------


# Ensemble 2: a caretList stack -------------------------------------------


# Ensemble 3: a tailor made ensemble averaging probs ----------------------
predProbs_2 <- avgListProbs(LC_modelsLists, trainingSet_2, "UNSATISFIED")

# Ensemble 4: tailor made ensembles fitting models from submodels  --------

# We'll use as predictors the probs predicted by single low correlated 
# submodels on trainingSet_2. The ensemble models obtained will be 
# Cross Validated on trainingSet_2 and validated on trainingSet_3.
set.seed(mySeed)

predictors_2 <- listProbs(LC_modelsLists, trainingSet_2, "UNSATISFIED")

modelsList2 <- caretList(# Arguments to pass to train() as '...' ###
  x          = predictors_2,
  y          = trainingClass_2, 
  preProc    = c("BoxCox"),
  metric     = "ROC",
  ###########################################
  # caretList() specific arguments ##########
  trControl  = trCtrl,
  methodList = classifModels,
  tuneList   = tuneModels
  ###########################################
)

save(modelsList2,  file = file.path(".", "Models", "modelsList22.Rda"))

# Compare Models 

## ON CV basis
resamps <- resamples(modelsList2)
summary(resamps)
modelsROC_CV <- summary(resamps)$statistics$ROC[,"Mean"]

xyplot(resamps, what = "BlandAltman")
dotplot(resamps)
densityplot(resamps)
bwplot(resamps)
splom(resamps)
parallelplot(resamps)

diffs <- diff(resamps)
summary(diffs)

listEnsembleCV_AUC(LC_modelsLists, modelsList2)

## ON validation set trainingSet_3
predictors_3 <- listProbs(LC_modelsLists, trainingSet_3, "UNSATISFIED")

modelsPerfVal <- modelsListPerformance(modelsList2, 
                                       predictors_3, trainingClass_3, 
                                       lev = c("SATISFIED", "UNSATISFIED"))

modelsROC_Val <- modelsPerfVal[,"ROC"]

listEnsembleAUC(LC_modelsLists, modelsList2,
                trainingSet_3, "UNSATISFIED",
                trainingClass_3)

## CV and Validation Set Methods - summary
summaryModelsPerf <- modelsListPerf_CV_VAL(modelsList2, 
                                           predictors_3, trainingClass_3, 
                                           lev = c("SATISFIED", "UNSATISFIED"))
summaryModelsPerf

# Making a submission

makeAnEnsembleSubmission(LC_modelsLists, modelsList2, "nn", 
                                      tidy_test, "ens1NN.csv")
# 0.677362


# Clean and Exit ----------------------------------------------------------

# Stop Parallel Processing
stopCluster(cl)