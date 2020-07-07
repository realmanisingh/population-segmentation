# Machine Learning Engineer Nanodegree
## Capstone Project
Mani Singh  
December 31st, 2050

## I. Definition

### Project Overview
In the marketing industry, there's an old saying that marketers waste half of their money. While this may be hyperbole, a study conducted by Rakuten Marketing reveals that marketers waste on average 26% of their budgets on ineffective channels and strategies.[^1] However, with advancements in artificial intelligence, marketers may be able to create better customer segments and accurately predict whether or not those exposed to their marketing campaign will become a customer.
[^1]: This is the first footnote.

In this project, I created a machine learning model that can predict whether or not a recipient of a marketing campaign will become a customer. I also used machine learning techniques to identify different customer segments along with the variables that best describe the segments. The models are trained using data from a mail-order sales company in Germany provided by Bertelsmann/Arvato and is accessed through Udacity.

### Problem Statement
The project can be split into two main parts: unsupervised learning and supervised learning. However, before working on each part, the following tasks must be performed:

1. Download the general population, customer, training, and test data from Udacity.
2. Clean the data so that it can be used for analysis and training machine learning models.
3. Perform data analysis to increase familiarity with the data and discover interesting trends

After these tasks are completed, the unsupervised learning portion of the project can be completed, which includes the following steps:

1. Reduce the dimensionality of the customer data and general population data.
2. Use the k-means clustering machine learning algorithm to create segments in the customer data and the general population data.
3. Create a visualization that shows the most influential principal components for every cluster and then relate those principal components back to the features in order to learn what features make up each cluster.

The last part of the project will be training a machine learning model that can accurately predict whether or not a recipient of a marketing campaign will become a customer. The tasks for this part include:

1. Balancing the dataset so that the model has enough positive observations to learn from.
2. Use cross validation to compare the generalized performance of many different models.
3. Perform hyperparameter tuning and feature selection for the best models

### Metrics
There will be three metrics used to evaluate the supervised learning model: accuracy, recall, and precision.

Accuracy is a common metric used for binary classification problems and it measures the percentage of observations that were correctly classified. 

$$accuracy = \frac{\text{true positives} + \text{true negatives}}{\text{number of observations}}$$

Recall is a metric that measures percentage of actual positives that were classified correctly. In this case, it would represent the percentage of recipients of the marketing campaigns who became customers that were classified correctly by the supervised model. In order to achieve a high recall, the goal is to create a model that accurately predicts potential customers therfore reducing the number of false negatives. From a business standpoint, false negatives are not ideal because that means that the model did not identify people who would have likely became customers after being exposed to an marketing campaign.

$$recall = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}$$

Precision is a metric that measures the percentage of observations classified as positive by the model that were actually positive. A  high precision would mean minimizing the number of false positives. For this problem, that means minimizing the number of people that are  incorrectly classified as becoming customers.

$$precision = \frac{\text{true positives}}{\text{true positives} + \text{false positives}}$$

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
The data provided by Bertelsmann/Arvato consists for four different datasets that each contain over 300 features. The largest of the datasets is the general population data (891,211 observations). The next largest dataset is the customer data (191,652 observations). The training data and test data both contain about 43,000 observations. 

The columns of every dataset are categorical with most of the categorical columns already encoded to a numerical scale. The data also contains many null values, like the 'CAMEO_DEUG_2015' that is missing 98979 observations. There were also three columns whose values were over 95% null. In addition to null values, there are a few columns that still contain strings instead of numbers so they will need to be encoded. 


In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
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

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>