#! /usr/bin/env python3
import math
import sys

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.
    
    Args:
      california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
    Returns:
      A DataFrame that contains the features to be used for the model, including
      synthetic features.
    """
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] /
        california_housing_dataframe["population"])
    return processed_features


def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
      california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
    Returns:
      A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
        california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
      input_features: The names of the numerical input features to use.
    Returns:
      A set of feature columns
    """ 
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x:((x - min_val) / scale) - 1.0)


def log_normalize(series):
    return series.apply(lambda x:math.log(x+1.0))


def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x:(min(max(x, clip_to_min), clip_to_max)))


def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x:(x - mean) / std_dv)


def binary_threshold(series, threshold):
    return series.apply(lambda x:(1 if x > threshold else 0))


def normalize_linear_scale(examples_dataframe):
    """Returns a version of the input `DataFrame` 
     that has all its features normalized linearly.
    """
    normalized_frame = pd.DataFrame()
    normalized_frame["latitude"] = linear_scale(examples_dataframe["latitude"])
    normalized_frame["longitude"] = linear_scale(examples_dataframe["longitude"])
    #normalized_frame["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
    #normalized_frame["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
    #normalized_frame["population"] = linear_scale(examples_dataframe["population"])

    #normalized_frame["total_bedrooms"] = log_normalize(examples_dataframe["total_bedrooms"])
    #normalized_frame["households"] = log_normalize(examples_dataframe["households"])
    #normalized_frame["median_income"] = log_normalize(examples_dataframe["median_income"])
    #normalized_frame["rooms_per_person"] = linear_scale(examples_dataframe["median_income"])
    return normalized_frame


def train_nn_regression_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network regression model.
    
    In addition to training, this function also prints training
    progress information, as well as a plot of the training and 
    validation loss over time.
  
    Args:
      my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.
      
    Returns:
      A tuple `(estimator, training_losses, validation_losses)`:
      estimator: the trained `DNNRegressor` object.
      training_losses: a `list` containing the training loss values taken during training.
      validation_losses: a `list` containing the validation loss values taken during training.
    """

    periods = 10
    steps_per_period = steps / periods
  
    # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer
    )
    
    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value"],
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                  training_targets["median_house_value"],
                                                  num_epochs=1,
                                                  shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                    validation_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("  Training model...")
    print("  RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        
        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("    period {0:2d} : {1:.2f}".format(period, training_root_mean_squared_error))

        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("  Model training finished.")
        
    print("  Final RMSE (on training data):  ", training_root_mean_squared_error)
    print("  Final RMSE (on validation data): ", validation_root_mean_squared_error)
    return dnn_regressor, training_rmse, validation_rmse


normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

# Choose the last 5000 (out of 17000) examples for validation.
training_targets =  preprocess_targets(california_housing_dataframe.head(12000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))


# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(normalized_training_examples.describe())
print("normalized_validation examples summary:")
display.display(normalized_validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

input("Hit Enter to continue")

hidden_units = [10, 10, 5, 5, 5]

print("train using gradient descent optimizer")
_, gd_training_losses, gd_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    steps=500,
    batch_size=50,
    hidden_units=hidden_units,
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)


print("train using adaptaive gradient descent optimizer (Adagrad)")
_, adagrad_training_losses, adagrad_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
    steps=500,
    batch_size=50,
    hidden_units=hidden_units,
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

print("train using stochastic optimizer (Adam)")
_, adam_training_losses, adam_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
    steps=500,
    batch_size=50,
    hidden_units=hidden_units,
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

plt.figure(1)
plt.ylabel("RMSE")
plt.xlabel("Periods")
plt.title("RMSE vs. Periods GradDesc")
plt.plot(gd_training_losses, label='grad descent training')
plt.plot(gd_validation_losses, label = 'grad descent valid')
plt.legend()
plt.draw()
plt.pause(1)

plt.figure(2)
plt.ylabel("RMSE")
plt.xlabel("Periods")
plt.title("RMSE vs. Perids Adagrad")
plt.plot(adagrad_training_losses, label='Adagrad training')
plt.plot(adagrad_validation_losses, label='Adagrad validation')
plt.legend()
plt.draw()
plt.pause(1)

plt.figure(3)
plt.xlabel("RMSE")
plt.ylabel("Periods")
plt.title("RMSE vs. Periods (Adam)")
plt.plot(adam_training_losses, label='Adam training')
plt.plot(adam_validation_losses, label='Adam validation')
plt.legend()
plt.draw()
plt.pause(10)
input("Hit Enter")

