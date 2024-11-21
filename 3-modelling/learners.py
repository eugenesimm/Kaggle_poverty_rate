from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import xgboost as xgb
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt


def generate_best_RF_model(train_A):
    X = X_train = train_A.drop(columns=['psu_hh_idcode', 'subjectivePoverty_rating'], axis='columns')
    y = train_A['subjectivePoverty_rating']
    
    # Define the parameter grid
    params = {
        'n_estimators': [200, 300, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [2, 5, 50],
        'min_samples_leaf': [35, 42, 50],
    }

    # Create the scorer
    # log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=params,
        scoring='neg_log_loss',
        cv=5,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )

    # Fit the grid search
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    print("Best Log Loss Score:", -grid_search.best_score_)
    
    # Return the best model
    return best_model


def generate_best_XGB_model(train_data):

    X_train = train_data.drop(columns=['psu_hh_idcode', 'subjectivePoverty_rating'], axis='columns')
    y_train = train_data['subjectivePoverty_rating'] - 1
    param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [1, 3, 5, 7],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.3, 0.5, 0.7, 0.9],
    'colsample_bytree': [0.4, 0.6, 0.8]
    }

    # Create the XGBoost model
    xgb_model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=101)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_log_loss',  # Use log loss as the evaluation metric
        cv=5,                    
        verbose=1,               
        n_jobs=-1                
    )

    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Log Loss Score:", -grid_search.best_score_)

    best_model_xgb = grid_search.best_estimator_
    return best_model_xgb



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

def build_nn_model(hp, n_features):
    """
    Build a neural network model for hyperparameter tuning.
    Args:
        hp: Hyperparameter object from KerasTuner.
    Returns:
        model: A compiled Keras Sequential model.
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32),
                    activation='relu',
                    input_shape=(n_features,)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Hidden layer
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32),
                    activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(10, activation='softmax'))  # 10 classes

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_best_NN_model(train_data):
    """
    Generate the best neural network model using KerasTuner for hyperparameter tuning.
    Args:
        train_data (DataFrame): Training data with features and target variable.
    Returns:
        best_model_nn: Trained neural network model with best parameters.
    """
    # Preprocess training data
    X_train = train_data.drop(columns=['psu_hh_idcode', 'subjectivePoverty_rating'], axis='columns').values
    y_train = train_data['subjectivePoverty_rating'].values

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # One-hot encode the target
    y_train_one_hot = to_categorical(y_train - 1)  # Adjust target to 0-indexed for categorical encoding

    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_one_hot, test_size=0.2, random_state=42, stratify=y_train
    )

    # Initialize Keras Tuner
    tuner = kt.Hyperband(
        build_nn_model,
        objective='val_loss',  # Optimize for log loss
        max_epochs=50,
        factor=3,
        directory='kt_hyperband',  # Directory for storing tuning logs
        project_name='nn_hyperparam_tuning'
    )

    # Perform hyperparameter search
    tuner.search(X_train_split, y_train_split,
                 validation_data=(X_val, y_val),
                 epochs=50,
                 batch_size=32,
                 verbose=1)

    # Get the best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    print(f"  Units Layer 1: {best_hps.get('units_1')}")
    print(f"  Dropout Layer 1: {best_hps.get('dropout_1')}")
    print(f"  Units Layer 2: {best_hps.get('units_2')}")
    print(f"  Dropout Layer 2: {best_hps.get('dropout_2')}")
    print(f"  Learning Rate: {best_hps.get('learning_rate')}")

    # Train the best model
    best_model_nn = tuner.hypermodel.build(best_hps)
    best_model_nn.fit(X_train_split, y_train_split,
                      validation_data=(X_val, y_val),
                      epochs=50,
                      batch_size=32,
                      verbose=1)

    # Evaluate the best model
    val_loss, val_accuracy = best_model_nn.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return best_model_nn

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
    train_B_X = encoder(train_B_X)
    train_B_X = encode_filler(train_B_X)
    preds_proba = model.predict_proba(train_B_X)
    output_df = pd.DataFrame(preds_proba, columns=[f'subjective_poverty{i+1}' for i in range(preds_proba.shape[1])])
    output_df.insert(0, 'psu_hh_idcode', test_ids.values)  # Insert the ID column at the start
    return output_df

