how you market to a segment will vary  - online ads vs mail promos
find segmentations that were previously unknown - benchmark


# Machine Learning Engineer Nanodegree
## Market Segmentation Capstone Proposal
Mani Singh  
June 01, 2020

### Domain Background
Market segmentation is the process of dividing customers in a given market into groups based on common attributes. This is significant in marketing because it allows for personalization and more accurate targeting in advertising campaigns. Personalization is staritng to become necessary in marketing with 74% of customers expressing frustration when website content is not personalized.(https://www.business2community.com/marketing/10-surprising-stats-personalization-01791432#M7UP7gMEHj4QOfXm.97). Moreover, 59% of customers say that personalization influences their purchasing decision.(https://www.evergage.com/blog/consumers-want-personalization-stats-roundup/). Instead of a generic, one size fits all offer, the statistics show that marketing should harness the power of personalization to provide targeted advertising to customers. 

The idea of segmentation in marketing is not a new concept. In the 1600s, retailers would segment their customers by wealth. For example, shop owners would hold private showcases of goods in their home for wealthier customers. Also, in the late 1800s, Germany toy manufactuers were using geographic segmentation to produce toy models for specific countries, like American locomotives intended to be sold in the United States.(https://en.wikipedia.org/wiki/Market_segmentation). Segmentation today is also done based on demographics and geography but has adapted to also include behavior and the customer's journey with respect to the buying process. 

### Problem Statement
Creating accurate and useful customer segments can be a daunting task. To create customer segments, a lot of reserach is required and a company may find that they need to conduct a study. This process can be time consuming and expensive. Often times, marketers are able to create simple segments, for example segmenting by country, age, gender, etc. But these segmentations are simple and do not contribute to understanding different segments of customers. 

### Datasets and Inputs
There are four data files that will be used in this project and they were all provided by Arvato Financial Solutions. The "Udacity_AZDIAS_052018.csv" file contains demographic information for the general population of Germany and has 891211 rows and 366 features. The "Udacity_CUSTOMERS_052018.csv" file contains data for the customers of a mail-order company and has 191652 rows and 369 features. This dataset has 3 extra features which provide details about the customers. The last two datasets, "Udacity_MAILOUT_052018_TRAIN.csv" and "Udacity_MAILOUT_052018_TEST.csv", contain demographic information about people who were targets of a marketing campaign. They have 42982 and 42833 rows respectively. The TRAIN file has 367 features while the TEST file has 366 due to the fact that the outcome variable, whether or not the recipient became a customer of the company, is left out of the TEST file so proper evaluation can be performed. 

Collectively, the features of all four datasets are categorical. The files "Udacity_AZDIAS_052018.csv" and "Udacity_CUSTOMERS_052018.csv" will be used in unsupervised learning methods in order to create customer segments. These two datasets are very feature-rich which means the customer segments will give insight into the different types of groups that are customers of the company as well as their similarities with the general population. The last two files will be used to create a supervised learning model based on the customer segments. Specifically, the "Udacity_MAILOUT_052018_TRAIN.csv" will be used to train the model while the "Udacity_MAILOUT_052018_TEST.csv" will be used to test the model. 

### Solution Statement
In order to solve this problem, the first task will be to fit the CUSTOMER data to a clustering model, like K-means. This will yield customer grouping or segments. Clustering will also be used on the AZDIAS data, which fill find groupings for the general population of Germany. Then after analyzing the similarities between these two groupings, a supervised model, like logistic regression, can be used to make predictions. It will be trained on the TRAIN data and evaluated on the TEST data.

### Benchmark Model
The benchmark model that will be used is logistic regression. Logistic regression is a fairly simplistic, supervised classifier that can be used to predict whether or not a person becomes a customer. Classification problems can be measured using various metrics like, accuracy, precision, and recall. However, for the benchmark model, just accuracy will be used. Logistic regression models can be prone to overfitting so it is expected that the final supervised model chosen will outperform the benchmark model.

### Evaluation Metrics
As mentioned previously, the evaluation metric that will be used is accuracy. Accuracy will be a percentage and has the formula $\frac{\text{Number of Correct Predictions}}{\text{Number of Predictions}}$. The outcome variable of the supervised model is whether or not a person becomes a customer, which can be encoded with 1 or 0, where "1" is the person becomes a customer and "0" is they do not become a customer. Calculating the number of correct predictions would simply be getting the equality between the predicted vector and the actual vector. The number of predictions can be calculated by getting the length of the prediction vector. 

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?