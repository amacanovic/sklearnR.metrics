% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/performance_scores_multiclass.R
\name{performance_scores_multiclass}
\alias{performance_scores_multiclass}
\title{Calculate the (sklearn) multi-class F-score and accuracy statistics}
\usage{
performance_scores_multiclass(truth, prediction)
}
\arguments{
\item{truth}{a \code{vector} of true labels; it should take numerical
values only}

\item{prediction}{a \code{vector} of predicted labels; it should take numerical
values only}
}
\value{
A dataframe with the following measures: 

1. Macro F-score (calculated as the
average of F-scores of individual labels); corresponds to sklearn.metrics.f1_score() with parameter 'average'
set to 'macro' and the yardstick R package f_meas_vec with parameter 'estimate' set to 'macro';

2. Weighted F-score (calculated as the average of
F-scores of individual labels weighted by label frequencies);
corresponds to sklearn.metrics.f1_score with parameter 'average'
set to 'weighted';

3. Micro F-score / accuracy (calculated as the F-score using precision and
recall averaged over all categories), corresponds to sklearn.metrics.f1_score with parameter 'average'
set to 'micro' and the yardstick R package f_meas_vec with parameter 'estimate' set to 'micro';
}
\description{
A function to calculate different F-score and accuracy measures 
for multi-class classification problems. 
The measures here produce the same results as those obtained by the
python scikit package.
}
\examples{

truth <- c(0, 1, 2, 0, 1, 2)

prediction <- c(0, 2, 1, 0, 0, 1)

performance_scores_multiclass(truth, prediction)

  macro_f_score weighted_f_score micro_f_score_acc
      0.2666667        0.2666667         0.3333333
}
