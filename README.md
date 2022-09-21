sklearnR.metrics
================

<br>

This package calculates a range of different F-scores and accuracy
scores for multi-class and multi-label classification problems in R.

The goal of this package is to simplify the calculation of multi-class
and multi-label classification measures in R. Currently, different R
packages implement different types of averaging for measure calculation.
Especially when it comes to multi-label problems, this can cause issues
with interpretation and comparability of results across different
applications. In the documentation, we clearly denote which metrics
match those output by the sklearn.metrics functions - which are most
widely used in implementations using python - but also those given by
several R packages that offer options for multi-class (yardstick
package) or multi-label (HEMDAG and mlr) classification performance
measure calculation.

This package currently supports a variety of F-score (F1) measures for
multi-class and multi-label classification, as well as the accuracy and
subset-accuracy scores. More measures are planned to be added; please
share your suggestions on measures to implement in “Issues”.

## Installing and using the package

To install the package from github, use the following command:

``` r
devtools::install_github("amacanovic/sklearnR.metrics")
```

After installing, you will be able to load the package:

``` r
library(sklearnR.metrics)
```

## Multi-label measures

The main functionality of this package is the integration of multiple
types of multi-label F-scores and accuracy measures. It includes
functions comparable to those in sklearn, but also packages in R, such
as HEMDAG and mlr.

The function `performance_scores_multilabel` takes matrices with
observations in rows and labels in columns as follows:

``` r
truth_matrix <- matrix(c(0, 1, 0, 0, 1, 1, 0, 1, 1), nrow = 3, ncol = 3)

prediction_matrix  <-  matrix(c(0, 1, 1, 0, 1, 1, 0, 1, 0), nrow = 3, ncol = 3)
```

For clarification, in this example, the first row of the truth matrix
denotes the first observation, with the first label being negative, the
second positive, and the third negative again:

``` r
(truth_matrix)
```

    ##      [,1] [,2] [,3]
    ## [1,]    0    0    0
    ## [2,]    1    1    1
    ## [3,]    0    1    1

The function takes the truth and prediction matrices as input (allowing
only for values 1 and 0, the former denoting presence of a label, the
latter its absence), but also requires you to specify two parameters:

1.  the referece category against which the F-score is to be calculated
    (1 is the presence of the label, 0 is the absence of the label)

2.  the set of measures to be output by the function; this parameter can
    be set to `both` for all measures, `label_wise` for measures
    calculated per label only, and `observation_wise` for measures
    calculated per observation only:

``` r
performances <- performance_scores_multilabel(truth = truth_matrix,
                                              prediction = prediction_matrix,  
                                              reference_category = 1,  
                                              metrics = "both")
```

<br>

This function outputs a set of measures in a dataframe as shown below:
<br>

<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
macro_f\_score
</th>
<th style="text-align:right;">
weighted_f\_score
</th>
<th style="text-align:right;">
f_score_of_averages
</th>
<th style="text-align:right;">
micro_f\_score
</th>
<th style="text-align:right;">
accuracy
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
0.7777778
</td>
<td style="text-align:right;">
0.8
</td>
<td style="text-align:right;">
0.8333333
</td>
<td style="text-align:right;">
0.8
</td>
<td style="text-align:right;">
0.7777778
</td>
</tr>
</tbody>
</table>
<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
observation_micro_f\_score
</th>
<th style="text-align:right;">
average_observation_micro_f\_score
</th>
<th style="text-align:right;">
observation_f\_score
</th>
<th style="text-align:right;">
subset_accuracy
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
0.8333333
</td>
<td style="text-align:right;">
0.5
</td>
<td style="text-align:right;">
0.5
</td>
<td style="text-align:right;">
0.6666667
</td>
</tr>
</tbody>
</table>

<br> **The measures are as follows:** <br>

-   Label-wise metrics: these metrics are calculated either per
    individual label and then averaged; or are calculated across all

**1. Macro F-score** (average of F-scores for each label); corresponds
to
[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
with parameter ‘average’ set to ‘macro’;

**2. Weighted F-score** (average of F-scores for each label weighted by
label frequencies); corresponds to
[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
with parameter ‘average’ set to ‘weighted’;

**3. F-score of averages** (harmonic mean of average precision and
recall scores per label), as suggested
[here](https://arxiv.org/pdf/1911.03347.pdf);

**4. Micro F-score** (calculated using total true positives, false
negatives and false positives across all labels); corresponds to
[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
with parameter ‘average’ set to ‘micro’;

**5. Accuracy** (calculated as the percentage of correctly predicted
categories across all labels);

-   Observation-wise metrics: these metrics are calculated per
    observation, and then averaged for the whole dataset:

**6. Observation-based micro f-score** (harmonic mean of precision and
recall calculated per observation); corresponds to the output of the R
mlr package
[multilabel.f1](https://mlr.mlr-org.com/articles/tutorial/measures.html)
score;

**7. Average observation-based micro f-score** (harmonic mean of average
precision and recall scores per observation); corresponds to F score
output of HEMDAG R package
[F.measure.multilabel](https://search.r-project.org/CRAN/refmans/HEMDAG/html/multilabel.F.measure.html)
function with b.per.example set to ‘FALSE’;

**8. Observation-based F-score** (average of F-scores for each
observation); corresponds to
[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
with parameter ‘average’ set to ‘samples’ and avF score output of HEMDAG
R package
[F.measure.multilabel](https://search.r-project.org/CRAN/refmans/HEMDAG/html/multilabel.F.measure.html)
function with b.per.example set to ‘FALSE’;

**9. Subset accuracy** (calculated as the percentage of observations
where all labels were correctly predicted); corresponds to
[sklearn.metrics.accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
for multi-label problems.

## Multi-class measures

This package also includes a function `performance_scores_multiclass`
that calculates scores comparable to those obtained by sklearn’s
[f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
and
[accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
measures, as well as the function
[f_meas_vec](https://yardstick.tidymodels.org/reference/f_meas.html)
from the yardstick R package.

This function takes numerical vectors as input, e.g.:

``` r
truth_vector <- c(0, 1, 2, 0, 1, 2)

prediction_vector <- c(0, 2, 1, 0, 0, 1)

performances  <-  performance_scores_multiclass(truth = truth_vector, 
                                                prediction = prediction_vector)
```

<br> And outputs a dataframe with three measures as shown below:
<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
macro_f\_score
</th>
<th style="text-align:right;">
weighted_f\_score
</th>
<th style="text-align:right;">
micro_f\_score_acc
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
0.2666667
</td>
<td style="text-align:right;">
0.2666667
</td>
<td style="text-align:right;">
0.3333333
</td>
</tr>
</tbody>
</table>

<br> **The measures are as follows:** <br>

**1. Macro F-score** (calculated as the average of F-scores of
individual labels). This measure corresponds to
[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
with parameter ‘average’ set to ‘macro’ and the yardstick R package
[f_meas_vec](https://yardstick.tidymodels.org/reference/f_meas.html)
with parameter ‘estimate’ set to ‘macro’;

**2. Weighted F-score** (calculated as the average of F-scores of
individual labels weighted by label frequencies); corresponds to
[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
with parameter ‘average’ set to ‘weighted’;

**3. Micro F-score / accuracy** (calculated as the F-score using
precision and ‘recall averaged over all categories), corresponds to
[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
with parameter ’average’ ‘set to ’micro’ and the yardstick R package
[f_meas_vec](https://yardstick.tidymodels.org/reference/f_meas.html)
with parameter ‘estimate’ set to ‘micro’.
