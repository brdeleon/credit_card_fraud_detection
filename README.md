# Fraud Detection for Credit Card Transactions

## Preventing Financial Losses with Machine Learning
**Author**: [Brenda De Leon](mailto:brendardeleon@gmail.com)
<img src="https://campuscu.com/media/7470/card-theft2.jpg" alt="Campus USA Credit Union Photo Protecting Your Credit Card From Fraud" style="width: 750px;"/>

## Overview

E-commerce increased dramatically as a result of COVID-19 movement restrictions and fears. The convenience of online shopping continues to make consumers more reliant and retailers continue to find new ways to keep up with the demand. [Many small businesses](https://www.chargebackgurus.com/blog/contactless-payment-limits-increase) were pushed to enter the realm of e-commerce during the pandemic and are now reaching customers outside of their local area.

## Business Problem

Unfortunately, there has also been an increase in fraudulent transactions for smaller and larger retailers and with that an increase in financial losses. Digital card-not-present or contactless transactions provide criminals easier opportunities for fraud. Credit card fraud can result in financial losses for e-commerce businesses, including chargebacks and other related expenses, which can ultimately affect their profitability and sustainability. These issues can be expensive and frustrating and are possibly magnified for small businesses. [Experts say](https://www.cnbc.com/2021/01/27/credit-card-fraud-is-on-the-rise-due-to-covid-pandemic.html) there are not enough regulations to protect smaller businesses from the losses caused by fraudulent transactions.

E-commerce platforms such as [Shopify](https://www.shopify.com/) can protect their retailers from losses by incorporating data-based fraud prevention measures. 

Shopify has contracted us to build an accesible credit card fraud detection tool for their system to improve their sellers experience with their platform. 

We will provide Shopify with a machine learning model that can detect credit card fraud transactions by analyzing patterns to identify anomalies.


## Data

The credit card transaction dataset we used can be obtained [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?datasetId=310&sortBy=relevance). The dataset contains online transactions made in September 2013 over two days using European credit cards. The transactions are labeled as either fraudulent or non-fraudulent.

With the positive (fraudulent) class only representing 0.0017 of the cases, we will compare different methods to addressing the dataset's high imbalance. 

![distributionoftime](/distributionoftime.png)

<b>Data Understanding:</b> 
The primary role of this project is to learn about data imbalance. This dataset will allow me to explore techniques and tools to work with imbalanced data. 

## Methods

1. Load and Preprocess the data. Load data into a pandas DataFrame, preprocess it to remove duplicates, check for nulls, and detect outliers. Perform initial data exploration to get better understanding of data. 

2. Exploratory Data Analysis, we will visualize the data to get insights into the features. We will compare the distributions of each feature.
    
3. Data Split. Split data into training, validation, and test sets. Training set is used to train machine learning model, training set is used to evaluate performance of models during training, and the test set is used to evaluate final performance. It is important that the class distribution is preserved across the sets.
    
4. Build and Evaluate a Baseline Model. A baseline model will help establish a benchmark to compare the performance of other models. 

5. Iteratively Perform and Evaluate Imbalance Techniques. Investigate different algorithms and techniques to determine whether they should be part the final model. Evaluate the model's performance using various metrics, mainly recall  as well as F1 score and ROC-AUC. We will check that the model can detect fraud (true positives) without producing too many false positives.

6. Evaluate a Final Model on the Test Set. Fine tune hyper-parameters of the final model to optimize perofrmnace. Interpret final model to understand how it is making predictions and which features are most important. 

## Results

We built a <b>Gradient Boosting Classification</b> model that is able to classify an online credit card transaction as fraudulent or non-fradulent.
Our model provides credit card fraud detection with an accuracy of 99%, a macro recall score of 94%, recall score of 89% for the positive class, and a maco f1 score of 66%, only 8 fraud transactions were not detected and the number of false positives were minimized. 

![final model](/finalmodel.png)

`------------------------------------------------------------`<br>
`Final Model GBM CLASSIFICATION REPORT VALIDATION `<br>
`------------------------------------------------------------`<br>
`              precision    recall  f1-score   support`<br>
<br>
`           0       1.00      0.99      1.00     45320`<br>
`           1       0.20      0.89      0.33        76`<br>
<br>
`    accuracy                           0.99     45396`<br>
`   macro avg       0.60      0.94      0.66     45396`<br>
`weighted avg       1.00      0.99      1.00     45396`<br>


We have determined which features are most important in classifying a transaction:

`feature  importance` <br>
`     V15       0.747` <br>
`      V5       0.074` <br>
`     V13       0.040` <br>
`     V11       0.030` <br>
`      V9       0.019` <br>


## Conclusions

Shopify can use this classification model to detect fraudulent credit card transactions. 

Not only will our model prevent fraud transactions from going undetected, it will also provide Shopify and it's retailers with these additional benefits:

- <b>Building trust with retailers</b>
- <b>Protection from potential legal implications</b> 
- <b>Value of quick fraud detection</b>
- <b>Competitive advantage</b>

<img src="https://images.business.com/app/uploads/2022/08/01034530/shopify.png" alt="Shopify Logo" style="width: 500px;"/>

### Next Steps

  - More data could significantly improve our prediction ability.
  - Take a closer look at the features that showed possible multicolinearity.
  - Use example dependent cost sensitive learning to weigh the cost of misclassification based on the transaction amount. 
  - Use feature importance to improve our model, removing those features with lower scores. 

## For More Information

See the full analysis in the [Jupyter Notebook](</fraud_detection_modeling.ipynb>) or review this [presentation](</presentation.pdf>).

For additional info, contact Brenda De Leon at [brendardeleon@gmail.com](mailto:brendardeleon@gmail.com)

## Repository Structure

```
├── data
│   ├── creditcard.csv
│   └── clean_df.csv
├── fraud_detection_EDA.ipynb
├── fraud_detection_modeling.ipynb
├── presentation.pdf
└── README.md
```