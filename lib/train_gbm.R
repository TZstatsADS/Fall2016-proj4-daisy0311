###Train GBM model with different parameters

train_gbm <- function(dat_train, label_train){
  
  #remove 0 variation columns
  #zero.var.col <- which(apply(dat_train, 2, var)==0)
  #x <- dat_train[,-zero.var.col]
  
  #fit gbm model with selected parameters
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=2000,
                     distribution="poisson",
                     interaction.depth = 5, 
                     n.minobsinnode = 20, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  
  best_iter <- gbm.perf(fit_gbm, method = "OOB")
  
  return(list(fit=fit_gbm, iter=best_iter))
}

