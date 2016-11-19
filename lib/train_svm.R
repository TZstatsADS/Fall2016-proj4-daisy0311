###Train SVM model with different parameters

train_svm <- function(dat_train, label_train, par){
  
  if (is.null(par$gamma)){
    fit_svm <- svm(x=dat_train, y=label_train, kernel = par$kernel, 
                   cost = par$cost)
  } else{
    fit_svm <- svm(x=dat_train, y=label_train, kernel = par$kernel, 
                   cost = par$cost, gamma = par$gamma)
  }
  
  return(fit_svm)
}