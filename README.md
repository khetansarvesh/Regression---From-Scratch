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

The aim of the project is to implement multi linear regression and polynomial regression via different methods namely:
normal equations
gradient descent
stochastic gradient descent
gradient descent with ridge and lasso regularization
stochastic gradient descent with ridge and lasso regularization

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

#### Defining our Gradient Descent with Ridge Regularization Class:
This algorithm is the same as gradient descent just that the error function changes here, this change in error function is brought so that we can handle overfitting conditions.

What is overfitting?if you get good accuracy wrt train dataset but bad with test dataset then it is said that the model is overfitting the train dataset and hence we need to add some constraints to the weight vector.In ridge regression the constraint is

0nMi2<some eta value , where Mi represents weight

This condition can be imposed by changing the error function as stated below (results obtained from optimization theory)
E = sum of squared error function = 1/2*1n(y-y')2 + (lambda)0mMi2
Due to this change in error function derivative of the error function changes thus causing change in weight update formula given by
(Mi)new= (Mi)old - (α)*((∂ E / (Mi)old))

Vectorized formula derivation can be seen below


#### Defining our Stochastic Gradient Descent with Ridge Regularization Class:
Everything same as stochastic gradient descent just that as explained in above section to solve overfitting we add a new term in our error function and exactly same explanation goes here just that here since it is stochastic we will not be considering all the points to calculate error we will goes random point to calculate error in each iteration. Error function in this case is given by following 
E = sum of squared error function = 1/2*(yp-yp')2+(lambda)0mMi2
Due to this change in error function derivative of the error function changes thus causing change in weight update formula given by
(Mi)new= (Mi)old - (α)*((∂ E / (Mi)old))

Vectorized formula derivation can be seen below

#### Defining our Gradient Descent with Lasso Regularization Class:
Exactly same as the gradient descent with ridge regularization but in ridge regularization the constraint put on weight vector to solve overfitting problem is 0nMi2<some eta value while here we change this constraint and put other type of constraint that is given by 
0nMi<some eta value, where Mi represents weight
This condition can be imposed by changing the error function as stated below (results obtained from optimization theory)
E = sum of squared error function = 1/2*1n(y-y')2+ (lambda)0mMi
Due to this change in error function derivative of the error function changes thus causing change in weight update formula given by
(Mi)new= (Mi)old - (α)*((∂ E / (Mi)old))


#### Defining our Stochastic Gradient Descent with Lasso Regularization Class:
Exactly same as the stochastic gradient descent with ridge regularization but in ridge regularization the constraint put on weight vector to solve overfitting problem is 0nMi2<some eta value while here we change this constraint and put other type of constraint that is given by 
0nMi<some eta value, where Mi represents weight
This condition can be imposed by changing the error function as stated below (results obtained from optimization theory)
E = sum of squared error function = 1/2*(yp-yp')2+ (lambda)0mMi

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

**Q.What primary difference did you find between the plots of Gradient Descent and Stochastic Gradient Descent?**  

As you can clearly see that both gd and sgd does converge to the global minima but the convergence of gd is smooth i.e. the error always decreases while in sgd the convergence is not smooth i.e. the error might increase sometimes in middle but overall it decreases and ensures that it reaches the global minima gradually.This is due to the fact that in sgd we consider only one randomly selected data point to calculate error.You can refer to follow diagram to understand what exactly is happening in gd and sgd.

**Q.What happens to the training and testing error as polynomials of higher degree are used for prediction?**

In case of a small dataset as this as the degree of the polynomials increases, training error decreases because the model is trying to fit the training data and perfectly predict the training data but the same model might not predict the test data with same accuracy because it has fit the training data too perfectly and hence testing error increases.Such a scenario is called overfitting.

**Q.Does a single global minimum exist for Polynomial Regression as well? If yes, justify.**
Yes because the error/cost function is the same just the dataset has changed Yes,because we are ultimately generating new polynomial features and using the same error function which was earlier proved to be having a global minima.

**Which form of regularization curbs overfitting better in your case? Can you think of a case when Lasso regularization works better than Ridge?**

Ridge regularization had slightly better results compared to lasso regularization.Say you used no regularization and coefficients of a feature turned out to be very low like 0.000001 then we conclude that this feature is very less important and hence needs to be removed i.e. it can be removed by making it coefficient as 0. Now if we use ridge regularization this same feature coefficient turns out to be lower but ridge regularization does not reduces it to 0 but if we use lasso regularization then the coefficient of this feature becomes zero thus eliminating the least important feature and hence lasso is better in such a case.

**Q.How does the regularization parameter affect the regularization process and weights? What would happen if a higher value for λ (> 2) was chosen?**

Larger the regularization parameter means more penalizing of the larger weight features, hence, the extent of overfitting is inversely proportional to regularization parameter. This regularization parameter thus, helps to handle overfitting.

As lambda is increased, the regression model starts to underfit the data and slowly the weights of some features become insignificant (tend to zero). For a really high lambda (>100), the regression line is almost parallel to the x-axis since only theta zero significantly contributes to the equation. Testing and training errors are extremely high in such a case.

**Q.Regularization is necessary when you have a large number of features but limited training instances. Do you agree with this statement?**

Yes when there are low numbers of training data points the regression model tries to fit perfectly while sometimes might lead to overfitting.Overfitting is not desirable because it gives huge errors wrt testing data. Hence, a regularization term is introduced to penalize the large parameters which is the cause of overfitting.

**Q.If you are provided with D original features and are asked to generate new matured features of degree N, how many such new matured features will you be able to generate? Answer in terms of N and D.**

Number of features for (D,N)= (D+N)C(N) where D is the number of the original features, N is the degree of the polynomial.(not including bias term)

**Q.What is bias-variance trade off and how does it relate to overfitting and regularization.**

As we know that regularization is done to prevent overfitting and overfitting is nothing but low error wrt to train dataset and high error wrt test dataset.Error wrt to train dataset is also called bias while error wrt test dataset is also called variance.Hence to prevent overfitting you need to increase the error wrt to train dataset i.e. bias and decrease the error wrt test dataset i.e. variance.
The model with a high bias (training error) and high variance (testing error) faces a problem of under-fitting. Models with low bias and high Variance have to deal with overfitting problems. A good model would always have low Bias and low Variance. This is also called the Bias-Variance trade-off. The degree of the polynomial and the regularization parameters need to be tuned to arrive at the optimal bias-variance trade-off.


## **Teammembers :**

Rishabh Nahar 

Samkit Jain
