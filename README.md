# US-census-data-classification-task
This project shows how to classify data with numerical and categorical data, on an unbalanced dataset
The dataset contains about 300 000 rows and contains data about individuals , education work, family , race etc .
The goal is to predict whether the individual earns more than 50K dollars a year.
The dataset has about 6% individuals who earn more than 50K, thus making the dataset very unbalanced.
An important aspect is to have a good recall, as identifying most of the individuals with more than 50k earnings is important.
Thus, the ROC curves and setting threshold values for the classifier is an important aspect of the problem.

The choice of the classifier will be logistic regression, a simple supervised algorithm.This model estimates probabilities which will be helpful to set thresholds.The data does not need to be normalised.

We do a 10 fold cross validation to see the performance of our model.However we are choosing the f1 score as the metric because we have an unabalanced dataset with few positive observations.

The plots of correlation and scatterplots are used to choose which variables we will be keeping, and variables whith a high amount of NUlls are discarded. Categorical variables are transformed into dummy variables in order to be used by the model.

The performance of logistic regression is compared to a random forest classifier .The random forest does not predict probabilities but is helpful in order to explain the decision process when looking at the trees.
After fitting(threshold=0.5), the logistic regression yeilds 95% of accuracy,a precision of 73% and a recall of 0.38.The auc is 0.94 which is an indicator that the classifier is good.

After setting the threshold, we have a  maximum 0.57 f1 score on the train and test datasets , with a 0.26 threshold.

