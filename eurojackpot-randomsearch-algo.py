import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Load the data
data = pd.read_csv('C:\\Users\\karki\\Desktop\\eurojackpot-result-script\\euro-original-adjusted.csv')  # replace with your csv file path

# Standardize the 'Week' and 'Year' columns
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Week', 'Year']])

# Perform DBSCAN
dbscan = DBSCAN(eps=0.5)
dbscan.fit(data_scaled)

# Add the cluster labels to the adjusted DataFrame
data['Cluster'] = dbscan.labels_

# Create the target columns for the main numbers and extra numbers
for i in range(1, 51):
    data['main'+str(i)] = data[['no1', 'no2', 'no3', 'no4', 'no5']].apply(lambda x: i in x.values, axis=1)
for i in range(1, 13):
    data['extra'+str(i)] = data[['extra1', 'extra2']].apply(lambda x: i in x.values, axis=1)

# Define the features and the targets
features = ['Week', 'Year', 'Cluster']
main_targets = ['main'+str(i) for i in range(1, 51)]
extra_targets = ['extra'+str(i) for i in range(1, 13)]

# Create and train the XGBoost models with randomized search
param_distributions = {
    'n_estimators': randint(50, 150),
    'learning_rate': uniform(0.01, 0.1),
}

main_models = {}
for target in main_targets:
    model = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_distributions, n_iter=12)
    model.fit(data[features], data[target])
    main_models[target] = model.best_estimator_

extra_models = {}
for target in extra_targets:
    model = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_distributions, n_iter=12)
    model.fit(data[features], data[target])
    extra_models[target] = model.best_estimator_

# Predict the cluster for the next draw (assuming it's the first draw of week 35 in 2023)
next_draw_scaled = scaler.transform([[35.1, 2023]])
next_draw_cluster = dbscan.fit_predict(next_draw_scaled)

# Predict the likelihoods of the numbers for the next draw
next_draw_features = [next_draw_scaled[0][0], next_draw_scaled[0][1], next_draw_cluster[0]]
next_draw_features_df = pd.DataFrame([next_draw_features], columns=features)

main_probs = {}
for number, model in main_models.items():
    proba = model.predict_proba(next_draw_features_df)
    main_probs[number] = proba[0][1] if proba.shape[1] > 1 else 0

extra_probs = {}
for number, model in extra_models.items():
    proba = model.predict_proba(next_draw_features_df)
    extra_probs[number] = proba[0][1] if proba.shape[1] > 1 else 0

# Select the 5 main numbers and 2 extra numbers with the highest likelihoods
predicted_main_numbers = [int(label[4:]) for label in sorted(main_probs, key=main_probs.get, reverse=True)[:5]]
predicted_extra_numbers = [int(label[5:]) for label in sorted(extra_probs, key=extra_probs.get, reverse=True)[:2]]

print('Predicted main numbers:', predicted_main_numbers)
print('Predicted extra numbers:', predicted_extra_numbers)
