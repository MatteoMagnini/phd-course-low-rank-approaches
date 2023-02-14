# Ph.D. course "Low-Rank Approaches for Data Analysis" project

Project for the Ph.D course "Low rank approaches for data analysis"

## 1. Setup
To properly run the code, you need to install the packages listed in the file `requirements.txt`.
You can do this by running the following command in the terminal:

    pip install -r requirements.txt

The recommended version of python is 3.10.0 or above.

# 2. Dataset
The dataset used in this project is the [Taiwanese Bankruptcy Dataset](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction).
Unfortunately, it seems that UCI has some issues with their website, so the dataset is not available at the moment.
However, you can find it in the `data` folder.
Alternatively, it is available on Kaggle [here](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction).

## 2.1 Description
The dataset contains 6,348 observations and 95 features and one target variable.
The target variable is the `Bankrupt?` column, which is a binary variable.
The dataset is imbalanced, with 5,966 observations of non-bankrupt companies and 382 observations of bankrupt companies.

## 2.2 Analysis
A preliminary analysis of the dataset shows that there are no missing values.
The correlation matrix shows that there are some features that are highly correlated with each other.
Other features are not correlated at all with the target variable (or with other features).
In other words, these features do not provide additional information.
For example, the `Net Income Flag` feature is always equal to 1, so it does not provide any information.

![Correlation matrix](figures/correlation_matrix.svg)
*Correlation matrix of the original features*

![Covariance matrix](figures/covariance_matrix.svg)
*Covariance matrix of the original features*

## 2.3 Preprocessing
The preprocessing step consist in the removal of the features that are not correlated with the target variable.
The features are removed if their absolute correlation value with the target variable is less than 0.01.
After the preprocessing step, the dataset contains 71 features.
The correlation matrix after the preprocessing step is shown below.

![Correlation matrix](figures/correlation_matrix_post_processed.svg)
*Correlation matrix of the preprocessed features*

![Covariance matrix](figures/covariance_matrix_post_processed.svg)
*Covariance matrix of the preprocessed features*

## 2.3 More analysis
Computing the variance of the features shows that the majority of the features (62) have a variance that is less than the 1% of the total variance.
Three features have a variance that is greater than 1% and less than 10% of the total variance.
Six features have a variance that is greater than 10% of the total variance.
Considering these results, data space is reduced by PCA using the covariance matrix.

The following figures show the distributions of the 71 features.

![Distributions of features 1-16](figures/distributions/distributions_0.svg)
*Distributions of the first 16 features*

![Distributions of features 17-32](figures/distributions/distributions_1.svg)
*Distributions of the features 17-32*

![Distributions of features 33-48](figures/distributions/distributions_2.svg)
*Distributions of the features 33-48*

![Distributions of features 49-64](figures/distributions/distributions_3.svg)
*Distributions of the features 49-64*

![Distributions of features 65-71](figures/distributions/distributions_4.svg)
*Distributions of the features 65-71*

# 3. PCA

![PCA explained variance](figures/pca_variance.svg)
*PCA variance informative contribution: blue bars represent the percentage variance of the i-th principal component, while the red line represents the cumulative variance. Principal components whose percentage variance is less than 1% are not shown.*

The figure shows that the cumulative percentage variance is greater than 90% after the 6-th principal component.
Moreover, there is a clear elbow in the curve after the 6-th principal component.
Therefore, the data space is reduced to 6 dimensions.

## 3.1 Analysis

![PC1 and PC2 circle of correlation](figures/correlation_circle_0.svg)
*Circle of correlation of the first two principal components*

PC1 is positive correlated with features:
- `Quick Asset Turnover Rate`
- `Current Asset Turnover Rate`
- `Total Asset Turnover`

PC2 is positive correlated with features:
- `Total Asset Growth Rate`
- `Cash Turnover Rate`
- `Research and development expense rate`

![PC3 and PC4 circle of correlation](figures/correlation_circle_1.svg)
*Circle of correlation of the third and fourth principal components*

PC3 is positive correlated with features:
- `Cash Turnover Rate`

and negative correlated with features:
- `Total Asset Growth Rate`

PC4 is positive correlated with features:
- `Research and development expense rate`

![PC5 and PC6 circle of correlation](figures/correlation_circle_2.svg)
*Circle of correlation of the fifth and sixth principal components*

PC5 is positive correlated with features:
- `Fixed Assets Turnover Frequency`

and negative correlated with features:
- `Quick Assets/Total Assets`
- `Current Assets/Total Assets`

PC6 is positive correlated with features:
- `Current Asset Turnover Rate`

and negative correlated with features:
- `Quick Assets Turnover Rate`

