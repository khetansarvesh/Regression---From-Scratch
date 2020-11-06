# Supervised-ML-Library-Implementation

## **How to download this project :**
 1. Install Anaconda/Jupyter Notebook
 2. Install following dependencies/libraries - numpy , pandas , matplotlib
 3. Download all the given files and put them in a folder named “X” on the desktop
 4. open Anaconda Prompt and type <jupyter notebook>
 5. Desktop -> X -> Linear Regression
 6. Run the ipynb file

## **Aim of this project :**
In this project I have implemented algorithms such as normal equations, gradient descent, stochastic gradient descent, lasso regularization and ridge regularization from scratch and done linear as well as polynomial regression analysis on a 1338*4 dimension dataset which is available in the folder 'dataset.'
The aim of the project is to implement multi linear regression via three different methods namely:
normal equations
gradient descent
stochastic gradient descent

This assignment teaches us how actually the algorithms written in the sklearn library actually works and in depth mathematical intuition behind these algorithms.

The dataset used in this project consists of three independent features i.e. age(X1), bmi(X2) and number of children of an individual(X3) and a dependent feature insurance amount for that person(Y).Our goal is to use these independent features to predict the values of dependent feature for which we need to devise algorithms to learn the coefficients w.r.t. each independent variable.

Y = C + M1X1+ M2X2+M3X3

## **Code Design and Derivations :**

Following are the differenct ipynb in this project and in this section i will talk about what each of these files are coded for and their usage and the theory behind the same

The code can be divided into following subsections for clear understanding:

#### Importing Required Libraries:
You need to install and import numpy, pandas and matplotlib libraries to successfully run this project 

#### Defining Our Normal Equation Class:
Here once we get the error function we differentiate it w.r.t. all the weights and equate all the derivatives to zero because we want to reach the  global minima of the curve and at the global minima slope is zero i.e. derivative of the function is zero. Hence we need to solve the n+1 simultaneous equation for n independent + 1 bias variable.

To understand the math and derivation of the vectorized formula used in this class please refer to the following pictures.It is better to vectorize our algorithm rather than using a loop to iterate over all the data points because it makes our algorithm more efficient.

As you can see that since this method involves inversion of a matrix and inverting a matrix is computationally very expensive hence we required a more efficient method to reach the global minima and one of the way is gradient descent.One is advised to use this method if no of independent variables is 1000 or less, though this is not a strict condition but can be handy.(source-- Andrew Ng)


#### Defining Our Gradient Descent Class:
In this method we don't directly go to the minima but instead we reach close to the global minima of the cost/error function gradually by jumping from point to another point.This new point in every iteration is decided by the gradient at that old point. 

Initial / starting value of these weights can be taken any random value, but we have taken them as all zeros, it won't matter at what initial point one is starting because ultimately the goal is to converge to the global minima.

The weights are updated using following formula:
(Mi)new= (Mi)old - (α)*((∂ E / (Mi)old))
Where,
M = weight/coefficient			α = learning rate
 i = 0,1,.....no of weights
E = sum of squared error function = 1/2*1n(y-y')2
 Y’ represents predicted values while Y represents actual target values for n data points.

Vectorization of the algorithm and respective derivation of formula can be seen below:

The stopping criteria for our implementation is when change in error difference is less than 0.00000001 but if no of iterations specified by the user is less than the no of iterations taken to reach error difference less than 0.00000001 then the algorithm stops when the no of iterations is equal to the one defined by the user.

As you can see that since this method involves calculating error w.r.t all the datapoints, we can further implore the time taken to run the algorithm by considering only few data points, this is called mini batch gradient descent and if only 1 datapoint is considered then it is called stochastic gradient descent.One is advised to use this method if no of independent variables is 1000000 or less, though this is not a strict condition but can be handy.(source-- Andrew Ng)

#### Defining Our Stochastic Gradient Descent Class:
This algorithm is almost exactly the same as gradient descent but the only difference here is that while considering the error function in gradient descent we used error due to all points but in stochastic gradient descent we consider error only due to a single random point in the dataset.

Hence now the weights are updated using following formula:
(Mi)new= (Mi)old - (α)*((∂ E / (Mi)old))
Where,
M = weight/coefficient			α = learning rate
 i = 0,1,.....no of weights
E = sum of squared error function = 1/2*(yp-yp')2where ypand yp’ are actual target value of that chosen data point and predicted value of that data point respectively
The derivations for the vectorised formula can be understood from following image:


#### Defining Our Evaluation Metric Class:
This class is used to calculate the accuracy of our model.Now we know that there are various accuracy measures but here we have implemented three of them namely RMSE(root mean squared error) , MSE(mean squared error) and Total Error.Formula for each of these are given below.

RMSE = (1ny - y2)/n
MSE = (1ny - y2)/n
Total Error(SSRES) = 0.5*1ny - y2
For obvious reasons the lower the error i.e. closer to 0 the better is our model.All of these evaluation metrics are proportional to each other and hence can be used based on one's will.





## **Key points to ponder upon :**

**Do all three methods give the same/similar results? If yes, Why?**

Yes because ultimately we are wanting to converge to the same global minima just the way to converge has changes in fact in the normal equation we are not converging to the minima we are directing going to the minima.

**How does normalization/standardization help in working with the data?**

 Normalization helps to convert all the features to the same scale.But what is the issue with different scales?the issue is that say is feature 1 is of scale 1000 and feature 2 is of scale 0.1 then due to feature 1 you training time will increase.Thus if we scale feature 1 to same scale of feature 2 it will improve our training time.Now as a standard procedure we scale all features to values between -1 and 1 and best way to do this is converting distribution of that feature into a normal distribution, this is called normalization.Thus normalization helps in reducing the time taken by the algorithm to converge(training time).

**Does increasing the number of training iterations affect the loss?**

 Yes, however it is significant only in initial stages because when no of iterations are too large the reduction in error becomes negligible.This happens because ultimately we are going to reach close to the same global minima but after a certain amount of iterations we would have already reached very close to the global minima and increasing the no of iterations further would not have much effect cause slope near the global minima is tending to 0.

**What happens to the loss after a very large number of epochs (say, ~1010)**

  The change in loss will become negligible but not zero exactly because you cannot reach the exact global minima you always reach close to the global minima.

**Would the minima (minimum error achieved) have changed if the bias term (w0) were not to be used, i.e. the prediction is given as 𝑌 = 𝑊𝑇𝑋 instead of 𝑌 = 𝑊𝑇𝑋 + 𝐵.**

 Not much in this dataset because the coefficient of bias term is approx -0.002 which is negligible compared to other independent features coefficients but it might be in other scenarios like consider an example where actual target variable is non zero value while values of all independent features is zero in such a case if we did not use bias term then the predicted value will be zero causing a error while if we used bias then predicted value will be non zero thus reducing the error compared to non bias.Bias variable can be thought of as used for shifting the regression function.


**What does the weight vector after training signify?**

The weight vector signifies the coefficients or the importance of each independent feature to predict the dependent(target) feature or in other words how much each feature contributes to the prediction of target/dependent variable.

**Which feature, according to you, has the maximum influence on the target value (insurance amount)? Which feature has the minimum influence?**

The feature with highest coefficient/weight has the maximum importance which in this case is “ age” while the feature with minimum coefficient/weight has the minimum importance which in this case is “number of children ”. 

**Q. While generating matured features (polynomial features in this case), is it better to:
A) generate the matured (polynomial) features from the given features and then scale those obtained matured features 
                        OR
B) first scale the given features and then generate matured (polynomial) features from those scaled features.**

A.If you first scale down, your values usually lie between 0, 1 (normalise). Now if you generate matured features like x^20 (basically any high degree polynomial), those values become too small and almost become 0 in the standard precision of your PC. But if you generate the features and then scale, those very high values which were generated will now be scaled into (0,1). Values will be considerable to work with. So I recommend you all guys do this.

**Q.Why can't we think of lambda (regularisation parameter) as a trainable parameter instead of a hyperparameter and learn it as well during GD/SGD? Let's assume we are dealing with L2 regularisation (ridge) with initial lambda value as 0.5.What I am asking is why can't lambda be learnt like the other Weights**

A.The thing is is L is the L2 loss function, dL/d lambda will be sum of norms of W, which is always a positive value. Now this will cause the gradient descent to always pull the value of lambda down to 0 and later negative.So now if lambda is negative, you favour larger weights as you can see that the Loss term will decrease with larger weights. This beats the point of regularisation

## **Teammembers :**

Rishabh Nahar 

Samkit Jain
