# Credit Card Fraud Detection
**Author**: [Brenda De Leon](mailto:brendardeleon@gmail.com)

## Overview

E-commerce increased dramatically as a result of COVID-19 movement restrictions and fears. The convenience of online shopping continues to make consumers more reliant and retailers continue to find new ways to keep up with the demand. [Many small businesses](https://www.chargebackgurus.com/blog/contactless-payment-limits-increase) were pushed to enter the realm of e-commerce during the pandemic and are now reaching customers outside of their local area.

<img src="https://english.news.cn/20220316/aed3e20f331940c4b8c2b16c1f15b2e6/20220316aed3e20f331940c4b8c2b16c1f15b2e6_96e22deb2-5cd2-4a33-9335-c119411a9451.jpg.jpg" alt=" Image of Product Launch" style="width: 550px;"/>

## Business Problem

Unfortunately, there has also been an increase in fraudulent transactions for smaller and larger retailers and with that an increase in financial losses. Digital card-not-present or contactless transactions provide criminals easier opportunities for fraud. Credit card fraud can result in financial losses for e-commerce businesses, including chargebacks and other related expenses, which can ultimately affect their profitability and sustainability. These issues can be expensive and frustrating and are possibly magnified for small businesses. [Experts say](https://www.cnbc.com/2021/01/27/credit-card-fraud-is-on-the-rise-due-to-covid-pandemic.html) there are not enough regulations to protect smaller businesses from the losses caused by fraudulent transactions.

E-commerce platforms such as [Shopify](https://www.shopify.com/) can protect their retailers from losses by incorporating data-based fraud prevention measures. 

Shopify has contracted us to build an accesible credit card fraud detection tool for their system to improve their sellers experience with their platform. 

We will provide Shopify with a machine learning model that can detect credit card fraud transactions by analyzing patterns to identify anomalies.

## Data

The credit card transaction dataset we used can be obtained [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?datasetId=310&sortBy=relevance). The dataset contains online transactions made in September 2013 over two days using European credit cards. The transactions are labeled as either fraudulent or non-fraudulent.

<b>Data Understanding:</b> 
The primary role of this project is to learn about data imbalance. This dataset will allow me to explore techniques and tools to work with imbalanced data. 

***
## Methods

    1. Load and Preprocess the data. Load data into a pandas DataFrame, preprocess it to remove duplicates, check for nulls, and detect outliers. Perform initial data exploration to get better understanding of data. 

    2. Exploratory Data Analysis, we will visualize the data to get insights into the features. We will compare the distributions of each feature.
    
    3. Data Split. Split data into training, validation, and test sets. Training set is used to train machine learning model, training set is used to evaluate performance of models during training, and the test set is used to evaluate final performance. It is important that the class distribution is preserved across the sets.
    
    4. Build and Evaluate a Baseline Model. A baseline model will help establish a benchmark to compare the performance of other models. 

    5. Iteratively Perform and Evaluate Imbalance Techniques. Investigate different algorithms and techniques to determine whether they should be part the final model. Evaluate the model's performance using various metrics, mainly recall  as well as F1 score and ROC-AUC. We will check that the model can detect fraud (true positives) without producing too many false positives.

    6. Evaluate a Final Model on the Test Set. Fine tune hyper-parameters of the final model to optimize perofrmnace. Interpret final model to understand how it is making predictions and which features are most important. 

***

## Results

We built a <b>Gradient Boosting Classification</b> model that is able to classify an online credit card transaction as fraudulent or non-fradulent.

We can detect a fraudulent transaction at an accuracy of 88% and at an f1 macro score of 73%.

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

Google can use this classification model to identify the emotion of tweets about a particular topic, the topic could be past launches, new products, or the brand itself. With Google's upcoming launch, Google can analyze the words, phrases, and hashtags of past launches by sentiment to better understand the audience's reception to the launch to help shape the strategy for the new launch. Equally, Google can use this model during the launch for real time feedback and after the launch to analyze for feedback. By classifying tweets into sentiment classes, Google will be able to extract more meaningful patterns with the help of word clouds and graphs. 


![google negative cloud](/google%20negative%20word%20cloud.png)

<br>

![google positive graph](/googlepositivehashtag.png)

### Next Steps

 - Better data collection could significantly improve our prediction ability. We have an imbalanced dataset with majority "Neutral" sentiment values. More data, particularly for the minority classes could improve the model's performance. Additionally, some of the tweets were mislabeled, for next steps our model could benefit from training on more accurate labeled tweets.
 - Include new data by web scraping tweets so that we are able to collect usernames of tweet poster and so that our model is able to train on newer and larger data.
 - Use specific tweet tokenizer so that our model is able to handle emojis.
 - Engineer additional features like assigning a sentiment intensity score to each tweet using nltk' vader package.
 - Given the high weight the random forest algorithm gives to the hashtags feature, we should further inspect it for patterns.
 - Use feature importance to improve our model, removing those features with lower scores.

## For More Information

See the full analysis in the [Jupyter Notebook](</_Modeling.ipynb>) or review this [presentation](</_Presentation.pdf>).

For additional info, contact Brenda De Leon at [brendardeleon@gmail.com](mailto:brendardeleon@gmail.com)

## Repository Structure

```
├── data
│   ├── creditcard.csv
│   └── .csv
├── .ipynb
├── .ipynb
├── .pdf
└── README.md
```