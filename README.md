# Lottery Prediction Project
This project is an attempt to predict the outcomes of a lottery draw using machine learning techniques.

## Description
We use historical lottery data to train machine learning models to predict the numbers of future draws. The models consider the week and year of the draw and the cluster to which the draw belongs, as determined by a DBSCAN clustering algorithm.

We use XGBoost as our machine learning model, with hyperparameters optimized using random search.

## Installation
You need the following Python packages installed on your machine to run the code:

pandas,
sklearn,
xgboost
## You can install them using pip:


`pip install pandas sklearn xgboost`
## Usage
To run the code, download the Jupyter notebook and open it in Jupyter. The notebook contains the code and comments explaining each step.

Make sure you replace the file path in this line with the path to your own lottery data file:


`data = pd.read_csv('C:\\Users\\karki\\Desktop\\lottery-result\\results.csv')`

 You can run the entire notebook in one go, or run each cell individually. The final cell will print the predicted main numbers and extra numbers for the next draw.

## Disclaimer
This project is for educational purposes only. The predictions made by the models are not guaranteed to be accurate, and should not be used for actual lottery betting.

## License
This project is licensed under the terms of the MIT license.
