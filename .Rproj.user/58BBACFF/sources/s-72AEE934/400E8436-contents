#' Calculate the (sklearn) multi-label F-score and accuracy statistics 
#'
#' A function to calculate different F-score and accuracy measures 
#' for multi-label classification problems. 
#' This function produces the same results as those obtained by the
#' python scikit package, with the addition of a few more measures.
#'
#' @param truth a \code{matrix} with true labels, where labels are in columns
#' and cases are in rows; labels can only be 0 or 1
#' @param prediction a \code{matrix} with predicted labels, where labels are in columns
#' and cases are in rows; labels can only be 0 or 1
#' @param reference_category specifies which label to use as the 
#' reference to calculate the F-score measures; can take values \code{1} or \code{0}; default value is \code{1}
#' @param metrics specifies which metrics to output; can take values \code{"label_wise"}, \code{"observation_wise"}, or
#' \code{"both"}; default is \code{"both"}
#' 
#' @return A data-frame with the following measures:
#' 
#' - Label-wise metrics: these metrics are calculated either per individual
#' label and then averaged; or are calculated across all labels
#' 
#' 1. Macro F-score (average of F-scores for each label); corresponds to 
#' sklearn.metrics.f1_score() with parameter 'average' set to 'macro'; 
#'  
#' 2. Weighted F-score (average of F-scores for each label weighted by
#' label frequencies); corresponds to sklearn.metrics.f1_score with parameter 'average'
#' set to 'weighted';
#' 
#' 3. F-score of averages (harmonic mean of average precision and recall scores per label);
#' 
#' 4. Micro F-score (calculated using total true positives, false negatives and false positives
#' across all labels); corresponds to sklearn.metrics.f1_score with parameter 'micro'
#' set to 'weighted';
#' 
#' 5. Accuracy (calculated as the percentage of correctly predicted categories across all labels);
#' 
#' 
#' - Observation-wise metrics: these metrics are calculated per observation, and 
#' then averaged for the whole dataset:
#' 
#' 6. Observation-based micro f-score (harmonic mean of precision and recall
#' calculated per observation); corresponds to R mlr package 
#' 'multilabel.f1 score';
#' 
#' 7. Average observation-based micro f-score (harmonic mean of average precision
#' and recall scores per observation); corresponds to F score output of 
#' HEMDAG R package F.measure.multilabel function with b.per.example set to 'FALSE';
#' 
#' 8. Observation-based F-score (average of F-scores for each observation); corresponds to 
#' sklearn.metrics.f1_score with parameter 'average' set to 'samples' and avF 
#' score output of HEMDAG R package F.measure.multilabel function with b.per.example set to 'FALSE';
#' 
#' 9. Subset accuracy (calculated as the percentage of observations where all labels were correctly predicted);
#' corresponds to sklearn.metrics.accuracy for multi-label problems.
#' 
#' @examples 
#' truth <- matrix(c(0, 1, 0, 0, 1, 1, 0, 1, 1), nrow = 3, ncol = 3)
#' 
#' prediction <- matrix(c(0, 1, 1, 0, 1, 1, 0, 1, 0), nrow = 3, ncol = 3)
#' 
#' performance_scores_multilabel(truth, prediction, reference_category = 1, metrics = "both")
#' 
#' @export

performance_scores_multilabel <- function(truth,
                                prediction,
                                reference_category = 1,
                                metrics = "both"){
  
  if (! reference_category %in% c(0, 1)){
    stop("reference category can only be 1 or 0")}
  
  if (! metrics %in% c("label_wise", "observation_wise", "both")){
    stop("metrics parameter can only be set to label_wise, observation_wise, or both")}
  
  if(class(truth) != "matrix"){
    stop("true labels must be supplied in a matrix form")}
  
  if( class(prediction) != "matrix"){
    stop("predicted labels must be supplied in a matrix form")}
  
  if(!identical(dim(truth), dim(prediction))){
    stop("true and predicted matrices must have the same dimensions")}
  
  macro_f_vec <- numeric(length = ncol(truth))
  support_vec <- numeric(length = ncol(truth))
  recall_vec <- numeric(length = ncol(truth))
  precision_vec <- numeric(length = ncol(truth))
  tn_vec <- numeric(length = ncol(truth))
  tp_vec <- numeric(length = ncol(truth))
  fn_vec <- numeric(length = ncol(truth))
  fp_vec <- numeric(length = ncol(truth))

  
  recall_vec_obs <- numeric(length = nrow(truth))
  precision_vec_obs <- numeric(length = nrow(truth))
  f_obs_vec <- numeric(length = nrow(truth))
  
  truth_matrix <- truth
  prediction_matrix <- prediction
  
  if(!all(truth_matrix %in% c(0, 1))){
    stop("values in the truth matrix can only be 0 or 1")}
  
  if(!all(prediction_matrix %in% c(0, 1))){
    stop("values in the prediction matrix can only be 0 or 1")}
  
  if (reference_category == 1){
    one <- "1"
    zero <- "0"
  }else{
    one <- "0"
    zero <- "1"
  }
  
  # F score as a harmonic mean of total recall and precision
  # Matches the multilabel.f1 measure in mlr R package
  f_score_mlr_fun <- function(truth_matrix, prediction_matrix) {
    numerator = 2 * rowSums(truth_matrix & prediction_matrix)
    denominator = rowSums(truth_matrix + prediction_matrix)
    mean(ifelse(denominator == 0, 1, numerator / denominator))
  }
  
  obs_based_micro_f <- f_score_mlr_fun(truth_matrix, prediction_matrix)
  
  # get TP, TN, FP, FN per row of each matrix
  
  for (i in 1:nrow(truth)){
    
    truth_calc_obs <- truth[i, ] 
    if (length(levels(truth_calc_obs)) > 2){
      stop('more than 2 levels in the truth matrix')
    }
    truth_calc_obs <- factor(truth_calc_obs, levels = c(0, 1))
    prediction_calc_obs <- prediction[i, ]
    if (length(levels(prediction_calc_obs)) > 2){
      stop('more than 2 levels in the prediction matrix')
    }
    prediction_calc_obs <- factor(prediction_calc_obs, levels = c(0, 1))
    conf_matrix <- as.matrix(table(truth_calc_obs, 
                                   prediction_calc_obs))
    
    tn <- conf_matrix[zero, zero]
    tp <- conf_matrix[one, one]
    fn <- conf_matrix[one, zero]
    fp <- conf_matrix[zero, one]
    
    if(tp+fn == 0){
      recall <- 0
      # warning('sum of true positives and false negatives is 0')
    }else{
      recall <- tp / (tp+fn)
    }
    
    if(tp+fp == 0){
      precision <- 0
      # warning('sum of true and false positives is 0')
    }else{
      precision <- tp / (tp + fp)
    }
    
    if(isTRUE(precision == 0 & recall == 0)) {
      f_score <- 0
    }else{
      f_score <- (2 * precision * recall)/(precision + recall)
    }
    
    recall_vec_obs[i] <- recall
    precision_vec_obs[i] <- precision
    f_obs_vec[i] <- f_score
  }
  
  # F-score as the harmonic mean of precision and recall averaged per number of 
  # observations
  avg_obs_based_micro_f <- (2 * mean(precision_vec_obs) * mean(recall_vec_obs))/(mean(precision_vec_obs) + mean(recall_vec_obs))
  
  # F-score as the mean of F scores of each observation, this is the same as
  # scikit "samples" setting
  obs_based_f <- mean(f_obs_vec)
  
  for (i in 1:ncol(truth)){
    
    truth_calc <- truth[, i] 
    if (length(levels(truth_calc)) > 2){
      stop('more than 2 levels in the truth matrix')
    }
    truth_calc <- factor(truth_calc, levels = c(0, 1))
    prediction_calc <- prediction[, i]
    if (length(levels(prediction_calc)) > 2){
      stop('more than 2 levels in the prediction matrix')
    }
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
      # warning('sum of true positives and false negatives is 0')
    }else{
      recall <- tp / (tp+fn)
    }
    
    if(tp+fp == 0){
      precision <- 0
      # warning('sum of true and false positives is 0')
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
  
  # Macro F score is just an average of individual label scores
  
  macro_f <- mean(macro_f_vec)
  
  # Weighted F score is a weighted average of individual label scores
  weighted_f <- weighted.mean(macro_f_vec, support_vec)
  
  # Alternative macro F score (not recommended)
  # Reference: https://arxiv.org/pdf/1911.03347.pdf
  # This is the "F1 of averages" - harmonic mean of precision and recall
  # vectors of each label 
  f_averages <- (2 * mean(precision_vec) * mean(recall_vec))/(mean(precision_vec) + mean(recall_vec))
  
  # Micro F score calculates recall and precision over
  # all labels and only then outputs the F score
  recall_total <- sum(tp_vec) / (sum(tp_vec)+sum(fn_vec))
  precision_total <- sum(tp_vec) / (sum(tp_vec)+sum(fp_vec))
  micro_f <- (2 * precision_total * recall_total)/(precision_total + recall_total)
  
  
  # Accuracy over the whole dataset
  accuracy_total <- length(which(as.matrix(truth) == as.matrix(prediction)))/ (nrow(truth)*ncol(truth))
  
  # Subset accuracy (as in scikit) - exact matches
  j <-  1
  matching_rows <- numeric(length = nrow(truth))
  for (j in 1:nrow(truth)){
    match <- ifelse(sum(truth[j, ] == prediction[j, ]) == ncol(truth),
                    1, 
                    0)
    matching_rows[j] <- match
  }
  
  subset_accuracy <- length(which(matching_rows == 1))/length(matching_rows)
  
  if (metrics == "both"){
    
    measures <- cbind.data.frame(macro_f,
                                 weighted_f,
                                 f_averages,
                                 micro_f,
                                 accuracy_total,
                                 obs_based_micro_f,
                                 avg_obs_based_micro_f,
                                 obs_based_f,
                                 subset_accuracy,
                                 stringsAsFactors = FALSE)
    
    colnames(measures) <- c("macro_f_score",
                            "weighted_f_score",
                            "f_score_of_averages",
                            "micro_f_score",
                            "accuracy",
                            "observation_micro_f_score",
                            "average_observation_micro_f_score",
                            "observation_f_score",
                            "subset_accuracy")
    
  }
  
  if (metrics == "label_wise"){
    
    measures <- cbind.data.frame(macro_f,
                                 weighted_f,
                                 f_averages,
                                 micro_f,
                                 accuracy_total,
                                 stringsAsFactors = FALSE)
    
    colnames(measures) <- c("macro_f_score",
                            "weighted_f_score",
                            "f_score_of_averages",
                            "micro_f_score",
                            "accuracy")
    
  }
  
  if (metrics == "observation_wise"){
    
    measures <- cbind.data.frame(obs_based_micro_f,
                                 avg_obs_based_micro_f,
                                 obs_based_f,
                                 subset_accuracy,
                                 stringsAsFactors = FALSE)
    
    colnames(measures) <- c("observation_micro_f_score",
                            "average_observation_micro_f_score",
                            "observation_f_score",
                            "subset_accuracy")
    
  }
  return(measures)
}
