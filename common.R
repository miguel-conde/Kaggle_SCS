mySeed <- 1234
numCVs <- 3

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

ensembleProbs <- function (modelsList, modelEnsemble, newdata, levName) {
  model_preds <- lapply(modelsList, predict, 
                        newdata = newdata, type="prob")
  model_preds <- lapply(model_preds, function(x) x[,levName])
  model_preds <- data.frame(model_preds)
  ens_preds   <- predict(modelEnsemble, 
                         newdata = newdata, 
                         type="prob")
  model_preds$ensemble <- ens_preds
  model_preds
}