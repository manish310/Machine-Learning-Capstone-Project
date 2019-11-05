
# Machine Learning Engineer Nanodegree
## Capstone Project
Manish Kumar  
October 20th, 2019

## I. Definition

### Project Overview

This project is related to Movie Industry. Movie industry comprises of many uniques features in itself. To work with a dataset i.e. related to this industry is always going to be challenging, intriguing and involving a lot of fun. There are many interesting aspects in it and their influences on whole industry is pretty much evident. And this what we are going to do in this project and it is always more convincing to see the results through data rather than intuitively.
There are many research going on in this domain and many research papers already published. A paper published by Simonoff & Sparrow discusses various predictor variables that can affect the gross revenue of a movie e.g. Genre of the film, MPAA(Motion Picture Association of America) rating, origin country of the movie, star cast of the movie, production budget of the movie etc _[Ref.1]_ .

The project dataset under consideration is part of *Kaggle Competition Problem* (https://www.kaggle.com/c/tmdb-box-office-prediction). The dataset contains over 7000 movies, collected from The movie Database(TMDB). The task is to predict these movies overall box office revenue.

The dataset has columns like _budget, genres, overview, popularity, release_date, runtime, keywords, cast crew, revenue etc_. Out of 7398 movies, 3000 movies are given as training dataset and remaining 4398 movies as test dataset.   

### Problem Statement

In this project the task is to predict target variable (_Revenue_) using independent variables like _budget_, _genres_, _popularity_, _runtime_, _release date_, _cast_, _crew_ etc. This problem comes under supervised learning as we have labeled data. Further, This Supervised learning problem is about predicting the target (variable) so I need to build a regression analysis model. I need to train the model on the training data, then testing the model performance using test data and finally predicting the revenue for _unseen_ data. In Scikit-learn(The package I chosen to use) package provides many regression algorithms. By taking into account the nature of data I decided to use **SGDregressor** algorithm for this problem.
The Model will be used to predict gross _revenue_ of unseen data (4398 movies).
The final result will contain _movie_id_ and _revenue_ in csv file.

### Metrics

I have chosen R-squared score as the evaluation metrics of this project (**a regression analysis**).
R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination. R-squared value always lies between 0 and 1 _[Ref.2]_.

The most general definition of the coefficient of determination is $$(1-\frac{SS_{res}}{SS_{tot}})$$, where $$SS_{tot}=\sum_{i=0}^{n}(y_i-y^-)^2$$  is total sum of squares(proportional to the variance of the data) and $$SS_{res}= \sum_{i=0}^{n}(y_i-f_i)^2$$ is the sum of squares of residuals _[Ref.3]_.
A model with R-squared value of 0 means that the model is behaving like a naive predictor model(predict average value all the time) and definitely not a good one. However, a value of 1 indicated the model is predicting the target variable accurately. Keeping this criteria in mind my aim in this project to build a regression model with a R-squared score close to 1.

## II. Analysis

### Data Exploration

The training dataset contains 3000 rows and 23 columns and testing dataset contains 4398 rows and 22 columns. The datasets are a good blend of numerical and categorical columns. Some of the important categorical columns are _genres, original_language, production_companies, production_countries, release_date etc_. Similarly, some of the important numerical columns are _budget, popularity, runtime, revenue etc_. A snapshot of the training  dataset looks like below:

![](snapshot.png)

As we know that machine learning algorithms are susceptible to the input data. Hence, it becomes matter of utmost important to wrangle the data before feeding it into the model.

In data exploration step it is important to look for **missing values, duplicate values , null values and outliers** in the given data.
Using **describe()** function of Pandas I can easily see various statistics of all numerical features present in the dataset.

![](statistics.png)

The above chart shows statistics of 3 measure features and 1 target variable. We can see many abnormalities in the data. For variables _budget, popularity, runtime and revenue_ the minimum value is 0 and for revenue its 1. If we see maximum value all the variables have abnormally high values. This also signifies towards presence of outliers in the data. However, we can not say those records as outliers
without further investigation because the data comes from movies industry which has peculiar features in itself. For example some movies perform exceptionally well in term of gross revenue collection, similarly some movies are made with huge budget etc.

Null values are something that can break the learning model. For 'runtime' variable I found only two such records and I decided to remove those rows.

Floating point values also create problem for regression model so I converted variables like _budget, popularity, runtime and revenue_ to integer.
In original dataset columns is present in form of list of dictionary, so I need to convert this categorical variable in form of comma separated text so that later stage I can easily convert it into one-hot encoding.

![](genres.png)   

In the original dataset there is one columns 'release_date' which denotes the releasing date of the movie. This is in form of date data type. This columns is not of much of use for my model in this form, However new features can be created using this column like 'release_month', 'day_of_week' etc.

One of the main aspects of the data data exploration is to find out important features for the current problem from the available data. This can be done using univariate, bivariate and multivariate analysis of the data. These analysis I done using **Pandas, Seaborn and Matplotlib** libraries. The data visualization analysis is discussed in the next section.

### Exploratory Visualization

In data Exploratory analysis, first of all I will  discuss univariate analysis. My focus is on mainly 4 variables i.e _budget_, _runtime_, _popularity_ and _revenue_.
The below histogram shows distribution curve of these variables.

![](histogram.png)

It is evident from the graph that curve of variables _budget_, _revenue_, and _popularity_ are right-skewed distribution and distribution graph of _runtime_ is a normal distribution.
The right-skewed graph signifies a possibility of presence of outliers in case of _budget_, _revenue_, and _popularity_. A further analysis of the above variables shows that these are genuine records, so I have to leave them as it is.
The below bar chart shows budget of top ten movies respectively.

![](budget.png)

Also, The below bar chart shows revenue of top ten movies respectively.

![](revenue.png)

In bivariate analysis, my intention is to see the correlation between the features(_budget_, _runtime_, _popularity_) and the target variable(_revenue_).
The below graph shows the scatter-plots of the these important features against _revenue_.

![](scatter_plot.png)

From the above plots signifies a strong correlation between _budget_ and _revenue_. There is also correlation between _popularity_ and _revenue_, and _runtime_ and _revenue_ to some lesser extent.

We know that in regression model correlation between features and target variable is an important aspects to look upon while dong the EDA.
Therefore, By doing a multivariate analysis using heatmap we can see the correlation between different variables at one place in summarized form.
The below plots shows heatmap of the numerical variables of the dataset.

![](hit_map.png)

As discussed earlier, in above heatmap it is evident a strong correlation between  _budget_ and _revenue_. There is also correlation between _popularity_ and _revenue_, and _runtime_ and _revenue_.

### Algorithms and Techniques

In this project I used mainly two regression algorithm i.e. linear regression and stochastic gradient descent regression algorithm.
The simple linear regression algorithm establish linear relation between features and target variable.
The simple linear regression between feature _x_ and target variable _y_ is modeled as below equation:
$$y_i = b_0+b_1x_1$$  
Here, b_0 is called intercept i.e. expected value of target variable when the value of feature _x_ is 0.
b_1 is called slope i.e. the expected change in the target variable for each 1 unit increase in the feature variable _x_.
Similarly, when we try to predict value of target variable using multiple features then then it is called multiple linear regression. This can be modeled in form of equation as below:
$$y_i = b_0+b_1x_1+b_2x_2+b_3x_3+...$$.
The fitting of a regression line is done by minimizing the sum of the squared difference between true value and predicted value, $$(y-y_i) which is called residual.
The error or cost function is given as below:
$$SS_{res}= \frac{1}{2} \sum_{i=0}^{n}(y_i-f_i)^2$$
Different techniques can be used to minimize the cost function. One of them optimization algorithm is gradient descent.

We can use gradient descent to find the values of the model's parameters that minimize the value of the cost function. Gradient descent iteratively updates the value of the model's parameters by calculating the partial derivatives of the cost function at each step _[Ref.4]_.
Local minima problem can occur in case of gradient descent if the surface if non-convex. However, it works fine for the above cost function as it is convex in nature.
Learning rate is an important hyper-parameter of gradient descent. The gradient descent steps consist of subtracting the gradient of the error times the learning rate &aplha;.
If the learning rate is small enough, the cost function will decrease with each iteration until gradient descent has converged on optimal parameters.
If the learning rate is too large, the gradient descent could oscillate around the optimal values of parameters _[Ref.4]_.
There are two varieties of gradient descent: Batch gradient descent and Stochastic gradient descent.  
In this project I am using SGDRegressor of scikit-learn as the proposed model.

### Benchmark

For benchmark model, I used one of the simplest linear regression algorithm. This benchmark model will work as beseline for the problem.
```
from sklearn.linear_model import LinearRegression
reg= LinearRegression().fit(X,y) (where X is feature variables and y is target variable)
reg.Score(X,y)
```
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

## III. Methodology

### Data Preprocessing

 Data preprocessing is an important step in machine learning model building process. As I mentioned in the data exploration step that machine learning algorithms are very sensitive towards the input data. Hence, it becomes essential to pre-process the data for the better model performance.

 Also, It is important to note that categorical data in form of text breaks the machine learning algorithm, so the categorical data first needs to be converted into numerical data using one-hot encoding method.
 In Pandas software package there is a function get_dummies() using which I can create one-hot encoding very easily.
```
# Creating one-hot encoding for categorical variable 'release_month' and 'day_of_week'
import pandas as pd
features_1=pd.get_dummies(features)
```
```
# Creating one-hot encoding for the variable 'genres'
import pandas as pd
tmdb_train_df=pd.concat([tmdb_train_df,tmdb_train_df.genres.str.get_dummies(sep=',')],axis=1)
```
after doing one-hot encoding the dataset will look like this:
![](one_hot_encoding.png)

In this project my proposed model is SGDregressor from the scikit-learn package.

Stochastic gradient descent is sensitive to feature scaling, so it is highly recommended to scale the data. In this case, it is important to standardize the data(mean=o and variance=1). Also, the scaling to be done for training and testing features both to get the meaningful results[Ref.2].
Here I have used StandardScaler() function from the sklearn.preprocessing module to standardize the data.
```
# standardizing the testing and training features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
```

### Implementation

For this project to solve the underlying problem I built a regression model. The software package used is Scikit-learn as it provides a vast varieties of machine learning algorithms. For benchmark model, I used linear regression algorithm. This benchmark model will work as beseline for the problem.

**Benchmark Model:**
```
from sklearn.linear_model import LinearRegression
reg_b=LinearRegression()
reg_b.fit(X_train,y_train)
y_pred_b=reg_b.predict(X_test)
```
Here, I called *LinearRegression()* of sklearn.linear_model. This model is based on Ordinary Least Squares algorithm. *LinearRegression()* returns linear regression object **reg_b** using which we can fit the model.
As for the proposed solution for this project I chosen SGDregressor(Optimization of Linear Regression) algorithm seeing the nature of the dataset.
```
from sklearn import linear_model
reg=linear_model.SGDRegressor(max_iter=500)
reg.fit(X,y)
```


In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
I got output of the benchmark model and proposed model as shown below:

![](r3.png)

To further improve the performance of the model hyper hyperparmaters tuning is required. Hyperparameter tuning is one of the important aspects of model building process. An important hyperparameter of gradient descent is the learning rate. Other important hyperparameter in SGDregressor are _loss function, penalty, epsilon, n_iter, alpha etc_. These hyperparmaters need to be tuned to get the best performance that built in previous section.



In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
As discussed in the previous section, I have chosen R-squared score  as the evaluation metric for the model that I developed in the previous step. In sklearn, we have libraries using which r2_score can be calculated very easily.
```
from sklearn.metrics import r2_score
score=r2_score(y_true,y_predict)
```
By comparing this score value I can evaluate the model and further hyper parameter tuning can be done to improve the r2_score of the model.
This parameter tuning we are going to do in the section.
![](r1.png)
![](r2.png)
![](r4.png)
![](r5.png)


In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------
### References:

[1] Simonoff, Jeffrey S. and Sparrow, Ilana R., Predicting movie grosses: Winners and losers, blockbusters and sleepers, 1999 (https://archive.nyu.edu/handle/2451/14752)

[2] Regression Analysis: How Do I Interpret R-squared and Assess the Goodness-of-Fit?, 30 may 2013, (https://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit)

[3] https://en.wikipedia.org/wiki/Coefficient_of_determination

[4] Mastering Machine Learning with scikit-learn, 2014, Gavin Hackeling.

[5] Stochastic Gradient Descent, (https://scikit-learn.org/stable/modules/sgd.html)

[6] TMDB Box Office Prediction, Can you predict a movie's worldwide box office revenue?(https://www.kaggle.com/c/tmdb-box-office-prediction)




**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
