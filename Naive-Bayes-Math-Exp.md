# Introduction
Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

## Probabilistic Model
![Screen Shot 2021-06-18 at 4 48 04 AM](https://user-images.githubusercontent.com/73560826/122534284-6051b400-cff0-11eb-9642-5705720cff12.png)






### Constructing a classifier from the probability model
![Screen Shot 2021-06-18 at 4 52 18 AM](https://user-images.githubusercontent.com/73560826/122534873-f71e7080-cff0-11eb-95cf-318d2a7c1580.png)
