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


def preprocess_features(housing_dataframe):
    """Prepares input features from California housing data set.
  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
    selected_features = housing_dataframe[
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
        housing_dataframe["total_rooms"] /
        housing_dataframe["population"])


    processed_features["distance_from_san_francisco"] = pd.Series.abs(housing_dataframe["latitude"]-37)
    
    return processed_features


LATITUDE_RANGES = zip(range(32, 44), range(33, 45))


def select_and_transform_features(source_df):
    selected_examples = pd.DataFrame()
    selected_examples["median_income"] = source_df["median_income"]
    for r in LATITUDE_RANGES:
        selected_examples["latitude_{0:d}_to_{1:d}".format(r[0], r[1])] = source_df["latitude"].apply(
            lambda x: 1.0 if x >= r[0] and x < r[1] else 0.0)
    return selected_examples


def preprocess_targets(housing_dataframe):
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
        housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


def plot_corr(df, size=10):
    '''plots a graphical correlation matrix
    for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.draw()
    plt.pause(2)
    input("Hit Enter")


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
    input_features: The names of the numerical input features to use.
    Returns:
    A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1,
                shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated.
      None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model.

    In addition to training, this function also prints training
    progress information, as well as a plot of the training and
    validation loss over time.

    Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps.
    A training step consists of a forward and backward pass using a
    single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.

  Returns:
    A `LinearRegressor` object trained on the training data.
  """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                    training_targets["median_house_value"],
                                    batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets[
                                                "median_house_value"],
                                            num_epochs=1,
                                            shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                              validation_targets[
                                                  "median_house_value"],
                                              num_epochs=1,
                                              shuffle=False)

    # Train the model, but do so inside a loop so
    # that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0]
                                         for item in training_predictions])

        validation_predictions = linear_regressor.predict(
            input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0]
                                           for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(
                validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period {0:1d} : {1:.4f}".format(
            period, training_root_mean_squared_error))

        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.draw()
    plt.pause(10)
    input("Hit Enter")
    
    return linear_regressor


print("load data")
train_data_file = "california_housing_train.csv"
housing_dataframe = pd.read_csv(train_data_file, sep=",")

housing_dataframe = housing_dataframe.reindex(
    np.random.permutation(housing_dataframe.index))

# Choose the first 12000 (out of 17000) examples for training.
training_examples = select_and_transform_features(preprocess_features(housing_dataframe.head(12000)))
training_targets = preprocess_targets(housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = select_and_transform_features(preprocess_features(housing_dataframe.tail(5000)))
validation_targets = preprocess_targets(housing_dataframe.tail(5000))


correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]

corr = correlation_dataframe.corr()

# plot_corr(corr, 100)

# set features to use
minimal_features = ["median_income",
                    "distance_from_san_francisco",
                    ]

assert minimal_features, "You must select at least one feature!"

selected_training_examples = select_and_transform_features(training_examples)
selected_validation_examples = select_and_transform_features(validation_examples)

#plt.scatter(training_examples["latitude"],  training_targets["median_house_value"])
#plt.scatter(training_examples["distance_from_san_francisco"], training_targets["median_house_value"])
#plt.draw()
#plt.pause(1)
#input("Hit Enter")

learning_rate = float(sys.argv[1])
steps = int(sys.argv[2])
batch_size = int(sys.argv[3])
print("Parameters")
print("  learning rate {0}".format(learning_rate))
print("  steps {0}".format(steps))
print("  batch size {0}".format(batch_size))
train_model(
    learning_rate=learning_rate,
    steps=steps,
    batch_size=batch_size,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets)


