#' Calculate the (sklearn) multi-class F-score and accuracy statistics 
#'
#' A function to calculate different F-score and accuracy measures 
#' for multi-class classification problems. 
#' The measures here produce the same results as those obtained by the
#' python scikit package.
#'
#' @param truth a \code{vector} of true labels; it should take numerical
#' values only
#' @param prediction a \code{vector} of predicted labels; it should take numerical
#' values only
#' 
#' @return A dataframe with the following measures: 
#'
#' 1. Macro F-score (calculated as the
#'average of F-scores of individual labels); corresponds to sklearn.metrics.f1_score() with parameter 'average'
#'set to 'macro' and the yardstick R package f_meas_vec with parameter 'estimate' set to 'macro';
#'
#'2. Weighted F-score (calculated as the average of
#'F-scores of individual labels weighted by label frequencies);
#' corresponds to sklearn.metrics.f1_score with parameter 'average'
#'set to 'weighted';
#'
#'3. Micro F-score / accuracy (calculated as the F-score using precision and
#'recall averaged over all categories), corresponds to sklearn.metrics.f1_score with parameter 'average'
#'set to 'micro' and the yardstick R package f_meas_vec with parameter 'estimate' set to 'micro';
#'
#'
#'@examples
#'
#' truth <- c(0, 1, 2, 0, 1, 2)
#' 
#' prediction <- c(0, 2, 1, 0, 0, 1)
#' 
#' performance_scores_multiclass(truth, prediction)
#' 
#'   macro_f_score weighted_f_score micro_f_score_acc
#'       0.2666667        0.2666667         0.3333333
#' @export

performance_scores_multiclass <- function(truth,
                                prediction){
  
  n_classes_truth <- length(unique(truth))
  n_classes_pred <- length(unique(prediction))
  n_classes <- max(n_classes_truth, n_classes_pred)
  
  if (n_classes_truth == n_classes_pred | n_classes_truth > n_classes_pred){
    all_classes <- sort(unique(truth))
  } else {
    all_classes <- sort(unique(prediction)) }
  
  if(class(truth) != "numeric"){
    stop("true labels must be supplied in a form of a numeric vector")}
  
  if(class(prediction) != "numeric"){
    stop("predicted labels must be supplied in a form of a numeric vector")}
  
  if(!identical(length(truth), length(prediction))){
    stop("true and predicted vectors must have the same dimensions")}
  
  if(length(truth) < 2 | length(prediction) < 2){
    stop("please supply a vector of truth/prediction values longer than 1")
  }
  
  truth_matrix <- matrix(nrow = length(truth), ncol = n_classes)
  prediction_matrix <- matrix(nrow = length(prediction), ncol = n_classes)
  
  for (i in 1:ncol(truth_matrix)){
    current_class <- all_classes[i]
    
    truth_matrix[, i] <- ifelse(truth == current_class, 1, 0)
    prediction_matrix[, i] <- ifelse(prediction == current_class, 1, 0)
  }
  
  macro_f_vec <- numeric(length = ncol(truth_matrix))
  support_vec <- numeric(length = ncol(truth_matrix))
  recall_vec <- numeric(length = ncol(truth_matrix))
  precision_vec <- numeric(length = ncol(truth_matrix))
  tn_vec <- numeric(length = ncol(truth_matrix))
  tp_vec <- numeric(length = ncol(truth_matrix))
  fn_vec <- numeric(length = ncol(truth_matrix))
  fp_vec <- numeric(length = ncol(truth_matrix))
  
  one <- "1"
  zero <- "0"
  
  for (i in 1:ncol(truth_matrix)){
    
    truth_calc <- truth_matrix[, i] 
    truth_calc <- factor(truth_calc, levels = c(0, 1))
    prediction_calc <- prediction_matrix[, i]
    
    prediction_calc <- factor(prediction_calc, levels = c(0, 1))
    conf_matrix <- as.matrix(table(truth_calc, 
                                   prediction_calc))
    
    tn <- conf_matrix[zero, zero]
    tp <- conf_matrix[one, one]
    fn <- conf_matrix[one, zero]
    fp <- conf_matrix[zero, one]
    support <- tp + fn
    
    tn_vec[i] <- tn
    tp_vec[i] <- tp
    fn_vec[i] <- fn
    fp_vec[i] <- fp
    
    if(tp+fn == 0){
      recall <- 0
      warning('sum of true positives and false negatives is 0')
    }else{
      recall <- tp / (tp+fn)
    }
    
    if(tp+fp == 0){
      precision <- 0
      warning('sum of true and false positives is 0')
    }else{
      precision <- tp / (tp + fp)
    }
    
    if(isTRUE(precision == 0 & recall == 0)) {
      f_score <- 0
    }else{
      f_score <- (2 * precision * recall)/(precision + recall)
    }
    
    macro_f_vec[i] <- f_score
    recall_vec[i] <- recall
    precision_vec[i] <- precision
    support_vec[i] <- support
  }
  
  macro_f <- mean(macro_f_vec)
  
  # weighted F score is a weighted average of all categories
  weighted_f <- weighted.mean(macro_f_vec, support_vec)
  

  # Micro F score calculates recall and precision over
  # all labels and only then outputs the F score
  recall_total <- sum(tp_vec) / (sum(tp_vec)+sum(fn_vec))
  precision_total <- sum(tp_vec) / (sum(tp_vec)+sum(fp_vec))
  micro_f <- (2 * precision_total * recall_total)/(precision_total + recall_total)
  
  
  # accuracy
  accuracy_total <- length(which(truth == prediction))/length(truth)
  
  measures <- cbind.data.frame(macro_f,
                               weighted_f, 
                               micro_f,
                               stringsAsFactors = FALSE)
  
  colnames(measures) <- c("macro_f_score",
                          "weighted_f_score",
                          "micro_f_score_acc")
  
  return(measures)
}
