mySeed <- 1234
numCVs <- 3

# Make a submission
makeASubmission <- function (fitModel, fileName) {
  myPred <- predict(fitModel, tidy_test, type ="prob")
  submission <- data.frame(ID = tidy_test$ID, TARGET = myPred$UNSATISFIED)
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