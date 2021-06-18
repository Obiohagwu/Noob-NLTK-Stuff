import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# getting dataset
df = pd.read_csv("/content/mushrooms.csv")
# To print dimension. Should be (8124, 23): print(df.shape)
# To display table: print(df,head())

#In the above dataset, we see classes (y) denoted by the values 0 - 8124. We see the features(x) denoted by 23 items ranging from cap-shape to Habitat.

le = LabelEncoder()

#Apply works as a for loop. It will allow us to go through items in dataset, 
#while applying a given function to them

df_encoded = df.apply(le.fit_transform, axis=0)
#axis = 0 becuase we want iteration by each row

# To dislay the encoded version of the dataset with values switched to numerical version: print(df_encoded.head())

df = df_encoded.values

#Now we are going to start partitioning the dataset into X and Y

x = df[: , 1:]
y = df[:, 0]

# If needed: print("this is x: ", x)
# If needed: print("This is y: ",y)

x_train, y_train, x_test, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)


# If needed: print(y_test)

# REFER TO NAIVE BAYES ALGORITH MATH EXPLANATION .md FILE #

# We need to first get Likelihood and Prior probalility #


#Recall that prior probality P(y) is sum of elements of y divided by len(y)
def prior_probab(y_train, label):
  m = y_train.shape[0]
  s = np.sum(y_train == label)

  return s/m 


#Then we define the conditonal probality
def cond_probab(x_train, y_train, feature_col, feature_val, label):
  x_filtered = x_train[y_train == label]
  num = np.sum(x_filtered[:, feature_col] == feature_val)

  denom = x_filtered.shape[0]

  return float(num/denom)


def predict(x_train, y_train, x_test):

  classes = np.unique(y_train)
  n_features = x_train.shape[1]
  posterior_probab = [] #For every value of posterior probab, we get a percentage. Each percentage represents the prob of that word bein spam or not spam


  for label in classes:
    likelihood = 1.0
    for feature in range(n_features):
      cond = cond_probab(x_train, y_train, feature, x_test[feature], label)
      likelihood = likelihood * cond 
    prior = prior_probab(y_train, label)
    post = likelihood * prior 


    posterior_probab.append(post)

    #Argmax gives us the maximum value contained
    pred = np.argmax(posterior_probab)
    return pred 
  
  
  
#Now it's time to define a function to get accuracy

def accuracy(x_train, y_train, x_test, y_test):
  pred = [] #Containes the predicted value of each of the test values

  for i in range(x_test.shape[0]):
    p = predict(x_train, y_train, x_test[i])

    pred.append(p)
  y_pred = np.array(pred)
  acc = np.sum(y_pred == y_test)/y_pred.shape[0]

  return acc 

# This is buggy right now, will fix later... maybe?
acc = accuracy(x_train, y_train, x_test, y_test)



