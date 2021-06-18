# Introduction
Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

## Probabilistic Model
Abstractly, naïve Bayes is a conditional probability model: given a problem instance to be classified, represented by a vector ,![Screen Shot 2021-06-18 at 4 44 32 AM](https://user-images.githubusercontent.com/73560826/122533735-e28da880-cfef-11eb-96ee-b47e9272e636.png)



