from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from sklearn.svm import SVC
from sklearn.metrics import log_loss
import xgboost as xgb
import pandas as pd


def generate_best_RF_model(train_A):
    X = train_A.drop(columns=['psu_hh_idcode', 'subjectivePoverty_rating'], axis='columns')
    y = train_A['subjectivePoverty_rating']

    train_X, calib_X, train_y, calib_y = train_test_split(X, y, stratify=train_A['subjectivePoverty_rating'], test_size=0.2, random_state=42)

    # Define the parameter grid
    params = {
    'n_estimators': [100, 150, 200, 250],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 12, 15, 18, 20],
    'min_samples_split': [2, 8, 10, 12],
    'min_samples_leaf': [10, 15, 20, 30],
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=params,
        scoring='neg_log_loss',
        cv=3,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(train_X, train_y)
    best_rf_model = grid_search.best_estimator_

    # Log best parameters and uncalibrated log-loss
    print("Best Parameters:", grid_search.best_params_)
    print("Best Log Loss Score (Uncalibrated):", -grid_search.best_score_)

    # Calibrate the model on the holdout set
    print("Calibrating the Random Forest for better probabilities...")
    calibrated_rf_model = CalibratedClassifierCV(best_rf_model, method='isotonic', cv='prefit')
    calibrated_rf_model.fit(calib_X, calib_y)

    # Evaluate calibrated log-loss on calibration set
    calib_probas = calibrated_rf_model.predict_proba(calib_X)
    calib_loss = log_loss(calib_y, calib_probas)
    print("Calibrated Log Loss Score (on Calibration Set):", calib_loss)

    # Return the calibrated model
    return calibrated_rf_model


# def generate_best_RF_model(train_A):
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import RandomizedSearchCV
#     from sklearn.preprocessing import StandardScaler
    
#     # Prepare data
#     X = train_A.drop(columns=['psu_hh_idcode', 'subjectivePoverty_rating'], axis='columns')
#     y = train_A['subjectivePoverty_rating']
#     X_scaled = StandardScaler().fit_transform(X)
    
#     # Define parameter grid
#     params = {
#         'n_estimators': [200, 300, 500, 700],
#         'max_features': ['sqrt', 'log2', 0.2, 0.5],
#         'max_depth': [5, 10, 15],
#         'min_samples_split': [2, 10, 50],
#         'min_samples_leaf': [20, 30, 50, 70],
#         'bootstrap': [True, False]
#     }

#     # Initialize RandomizedSearchCV
#     random_search = RandomizedSearchCV(
#         estimator=RandomForestClassifier(class_weight='balanced'),
#         param_distributions=params,
#         n_iter=50,  # Randomly sample 50 combinations
#         scoring='neg_log_loss',
#         cv=5,
#         verbose=1,
#         n_jobs=-1,
#         return_train_score=True
#     )

#     # Fit the randomized search
#     random_search.fit(X_scaled, y)
#     best_model = random_search.best_estimator_

#     print("Best Parameters:", random_search.best_params_)
#     print("Best Log Loss Score:", -random_search.best_score_)
    
#     return best_model


def generate_best_XGB_model(train_A):

    X = train_A.drop(columns=['psu_hh_idcode', 'subjectivePoverty_rating'], axis='columns')
    y = train_A['subjectivePoverty_rating'] - 1

    train_X, calib_X, train_y, calib_y = train_test_split(X, y, stratify=train_A['subjectivePoverty_rating'], test_size=0.2, random_state=42)


    param_grid = {
    'learning_rate': [0.05, 0.1],        # Limit choices to the most common values
    'max_depth': [3, 5, 7],                # Focus on shallower trees for faster training
    'n_estimators': [100, 200, 250],         # Limit boosting rounds for quicker training
    'subsample': [0.7, 0.9],            # Common subsampling rates
    'colsample_bytree': [0.6, 0.8],     # Feature sampling
    'reg_alpha': [0, 0.1, 0.2],              # Light regularization
    'reg_lambda': [3, 5, 7]                # Moderate L2 regularization
    }

# Create the XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss',
        use_label_encoder=False, 
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_log_loss',  # Use log loss as the evaluation metric
        cv=3,                    
        verbose=1,               
        n_jobs=-1                
    )
    print("Starting GridSearchCV...")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV Completed...")

    best_model_xgb = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    print("Best Log Loss Score:", -grid_search.best_score_)

    # random_search = RandomizedSearchCV(
    #     estimator=xgb_model,
    #     param_distributions=param_grid,
    #     n_iter=50,  # Try 50 random combinations
    #     scoring='neg_log_loss',
    #     cv=2,
    #     verbose=1,
    #     n_jobs=-1
    # )
    # print("Starting RandomizedSearchCV...")
    # random_search.fit(X_train, y_train)
    # print("RandomizedSearchCV Completed...")
    # best_model_xgb = random_search.best_estimator_
    # print("Best Parameters:", random_search.best_params_)
    # print("Best Log Loss:", -random_search.best_score_)
    return best_model_xgb

    # Log best parameters and uncalibrated log-loss
    print("Best Parameters:", grid_search.best_params_)
    print("Best Log Loss Score (Uncalibrated):", -grid_search.best_score_)

    # Calibrate the model on the calibration set
    print("Calibrating the XGBoost model for better probabilities...")
    calibrated_xgb_model = CalibratedClassifierCV(best_xgb_model, method='sigmoid', cv='prefit')
    calibrated_xgb_model.fit(calib_X, calib_y)

    # Evaluate calibrated log-loss on calibration set
    calib_probas = calibrated_xgb_model.predict_proba(calib_X)
    calib_loss = log_loss(calib_y, calib_probas)
    print("Calibrated Log Loss Score (on Calibration Set):", calib_loss)


    # Return the calibrated model
    return calibrated_xgb_model


def generate_best_SVM_model(train_data):

    train_x = train_data.drop(columns=['psu_hh_idcode', 'subjectivePoverty_rating'])

    # Combine numerical and encoded categorical data
    processed_train = encoder(train_x)
    processed_train = encode_filler(processed_train)

    y = train_data['subjectivePoverty_rating']
    X = processed_train

  
    # Scale the features
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X)
    
   # GridSearch CV
    param_grid = {
        'C': [0.5, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }

    optimal_params = GridSearchCV(SVC(probability=True, random_state=42), param_grid, n_jobs=-1, cv=5, scoring='neg_log_loss')
    
    # Fit the model
    # optimal_params.fit(X_train_scaled, y)
    optimal_params.fit(X, y)
    best_model = optimal_params.best_estimator_

    print("Best Parameters:", optimal_params.best_params_)
    print("Best Log Loss Score:", -optimal_params.best_score_)
    return best_model

def predict_ratings_RF(model, train_B_X):
    test_ids = train_B_X['psu_hh_idcode']
    train_B_X = train_B_X.drop(columns=['psu_hh_idcode'])
    preds_proba = model.predict_proba(train_B_X)
    output_df = pd.DataFrame(preds_proba, columns=[f'subjective_poverty_{i+1}' for i in range(preds_proba.shape[1])])
    output_df.insert(0, 'psu_hh_idcode', test_ids.values)  # Insert the ID column at the start
    return output_df


def predict_ratings_XGB(model, train_B_X):
    test_ids = train_B_X['psu_hh_idcode']
    train_B_X = train_B_X.drop(columns=['psu_hh_idcode'])
    preds_proba = model.predict_proba(train_B_X)
    output_df = pd.DataFrame(preds_proba, columns=[f'subjective_poverty_{i+1}' for i in range(preds_proba.shape[1])])
    output_df.insert(0, 'psu_hh_idcode', test_ids.values)  # Insert the ID column at the start
    return output_df

def encoder(data):
  col = [col for col in data.columns if -1 in data[col].values]
  # One-hot encode categorical columns
  encoder = OneHotEncoder(sparse_output=False, drop=None)
  encoded = encoder.fit_transform(data[col])

  # Convert to DataFrame and combine with numerical features
  encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(col), index=data.index)
  numerical_df = data.drop(columns=col)

  # Combine numerical and encoded categorical data
  processed_df = pd.concat([numerical_df, encoded_df], axis=1)
  return processed_df


def encode_filler(data):
  na_col = ['Q06_11.0', 'Q11_5.0', 'Q06_10.0', 'Q11_9.0', 'Q06_9.0', 'Q07_-1.0', 'Q06_2.0', 'Q01', 'Q11_14.0', 'Q06_5.0',
              'Q11_2.0', 'Q06_4.0', 'Q08_-1.0', 'Q07_4.0', 'Q08_2.0', 'Q07_0.0', 'q02', 'Q06_8.0', 'Q07_2.0', 'Q11_3.0', 'Q03',
              'Q11_4.0', 'q23', 'Q11_7.0', 'Q11_13.0', 'Q06_1.0', 'Q19_2.0', 'Q06_-1.0', 'Q11_-1.0', 'Q11_10.0', 'q05', 'Q07_1.0',
              'Q11_12.0', 'Q19_-1.0', 'Q06_7.0', 'Q11_1.0', 'Q19_1.0', 'Q11_8.0', 'Q08_1.0', 'Q06_0.0', 'q03', 'Q06_3.0', 'q09',
              'Q07_3.0', 'Q11_11.0', 'Q06_6.0']

  for col in na_col:
      if col not in data.columns:
          data[col] = 0  # Assign 0
  
  filled_data = data[na_col]
  return filled_data


def predict_ratings_SVM(model, train_B_X):
    test_ids = train_B_X['psu_hh_idcode']
    preds_proba = model.predict_proba(train_B_X)
    # print(log_loss(train_B['subjectivePoverty_rating'], preds_proba))
    output_df = pd.DataFrame(preds_proba, columns=[f'subjective_poverty{i+1}' for i in range(preds_proba.shape[1])])
    output_df.insert(0, 'psu_hh_idcode', test_ids.values)  # Insert the ID column at the start
    return output_df
