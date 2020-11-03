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

## **Code Design :**

Following are the differenct ipynb in this project and in this section i will talk about what each of these files are coded for and their usage and the theory behind the same


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
