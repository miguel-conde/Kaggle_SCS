library(caret)
library(caretEnsemble)
library(doParallel)

mySeed <- 1234
numCVs <- 3
maxBP <- 5

# Make a submission
makeASubmission <- function (fitModel, fileName) {
  myPred <- predict(fitModel, tidy_test, type ="prob")
  submission <- data.frame(ID = tidy_test$ID, TARGET = myPred$UNSATISFIED)
#   fileSub <- file.path(".", "Submissions", fileName)
#   cat(sprintf("saving the submission file %s\n", fileSub))
#   write.csv(submission, fileSub, row.names = F)
  writeSubFile(fileName, submission)
}

writeSubFile <- function(fileName, submission) {
  fileSub <- file.path(".", "Submissions", fileName)
  cat(sprintf("saving the submission file %s\n", fileSub))
  write.csv(submission, fileSub, row.names = F)
}

modelPerformance <- function(model, predictors, reference, lev, model_name) {
  myData <- cbind(pred = predict(model, predictors), 
                  obs = reference, 
                  predict(model, predictors, type = "prob"))
  twoClassSummary(data = myData, lev = lev,
                  model = model_name)
}

modelsListPerformance <- function(modelsList, predictors, reference, lev) {
  kk <- lapply(modelsList, function(x) {
    modelPerformance(x, predictors, reference, 
                     lev = lev, model_name = x$name)
  })
  t(as.matrix(as.data.frame(kk)))
}

modelsListPerf_CV_VAL <- function(modelsList, predictors, reference, lev) {
  resamps <- resamples(modelsList)
  modelsROC_CV <- summary(resamps)$statistics$ROC[,"Mean"]
  
  modelsPerfVal <- modelsListPerformance(modelsList, 
                                         predictors, reference, lev)
  modelsROC_Val <- modelsPerfVal[,"ROC"]
  
  data.frame(CV = modelsROC_CV, Validation_Set = modelsROC_Val)
}

listProbs <- function(modelsList, newdata, levName) {
  model_preds <- lapply(modelsList, predict, 
                        newdata = newdata, type="prob")
  model_preds <- lapply(model_preds, function(x) x[,levName])
  data.frame(model_preds)
}

ensembleProbs <- function (modelsList, modelEnsemble, newdata, levName) {
  model_preds <- listProbs(modelsList, newdata, levName)
  ens_preds   <- predict(modelEnsemble, 
                         newdata = model_preds, 
                         type="prob")
  model_preds$ensemble <- ens_preds[,levName]
  model_preds
}

ensembleAUC <- function (modelsList, modelEnsemble, newdata, levName,
                             trainEnsembleClass, ...) {
  model_preds <- ensembleProbs(modelsList, modelEnsemble, newdata, levName)
  caTools::colAUC(model_preds, trainEnsembleClass, ...)
}

listEnsembleAUC <-function(modelsList, modelsEnsemble, newdata, levName,
                           trainEnsembleClass, ...) {
  tmp <- lapply(modelsEnsemble, function(x) {
    ensembleAUC(modelsList, x, newdata, levName,
                trainEnsembleClass, ...)
  })
  out <- as.data.frame(t(sapply(names(tmp), function(x) tmp[[x]])))
  colnames(out) <- colnames(tmp[[1]])
  out
}

listEnsembleCV_AUC <- function(modelsListLvl1, modelsListLvl2) {
  AUC_1 <- summary(resamples(modelsListLvl1))$statistics$ROC[,"Mean"]
  AUC_2 <- summary(resamples(modelsListLvl2))$statistics$ROC[,"Mean"]
  
  out <- matrix(AUC_1, 1, nrow = length(AUC_2), ncol = length(AUC_1))
  out <- as.data.frame(out)
  names(out) <- c(names(AUC_1))
  out$ensemble <- AUC_2
  rownames(out) <- names(AUC_2)
  out
}

avgListProbs <- function(modelsList, newdata, levName, ...) {
  predProbs <- listProbs(modelsList, newdata, levName)
  data.frame(AVG = apply(predProbs, 1, mean, ...))
}

getLowCorrModels <- function(modelsList, newdata, levName, 
                             cutoff = 0.75, printTable = FALSE) {
  model_preds <- listProbs(modelsList, newdata, levName)
  modCorr <- findCorrelation(cor(model_preds), cutoff = cutoff)
  lowCorModels <- names(modelsList)[-modCorr]
  if (printTable && length(lowCorModels) > 1) cor(model_preds[lowCorModels])
  lowCorModels
  
}

myEnsemblePredict <- function(modelsListLvl1, modelsListLvl2, modelName, 
                              newdata, type = "prob") {
  predLvl1Probs <- predict(modelsListLvl1, newdata = newdata)
  predLvl2 <- predict(modelsListLvl2[[modelName]], newdata = predLvl1Probs, 
                      type = type)
  
  predLvl2
}

# Make an ensemble submission
makeAnEnsembleSubmission <- function (modelsListLvl1, modelsListLvl2, 
                                      modelName, 
                                      newdata, fileName) {
  myPred <- myEnsemblePredict(modelsListLvl1, modelsListLvl2, 
                              modelName, newdata)
  submission <- data.frame(ID = tidy_test$ID, TARGET = myPred$UNSATISFIED)

  writeSubFile(fileName, submission)
}