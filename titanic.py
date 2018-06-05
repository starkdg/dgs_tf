#!/usr/bin/env python3

import pandas as pd
from IPython import display
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.data import Dataset
from tensorflow.python import debug as tf_debug
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.4f}'.format

#hooks = [tf_debug.LocalCLIDebugHook()]


def preprocess_features(dataframe):
    selected_features = dataframe[
        ['Pclass',
         'Sex',
         'Age',
         'SibSp',
         'Parch',
         'Fare',
         'Embarked']]
  
    processed_features = selected_features.copy()
    return processed_features


def preprocess_targets(dataframe):
    output_targets = pd.DataFrame()
    output_targets["Survived"] = dataframe["Survived"]
    return output_targets


def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries, interpolation='linear')
  return [quantiles[q] for q in quantiles.keys()]


def construct_feature_columns(training_examples):

    pclass_column = tf.feature_column.categorical_column_with_identity(
        'Pclass', 4, default_value=0)
    sex_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'Sex', vocabulary_list=('male', 'female'), num_oov_buckets=1)

    age_column = tf.feature_column.numeric_column("Age", default_value=0)
    sibsp_column = tf.feature_column.numeric_column("SibSp", default_value=0)
    parch_column = tf.feature_column.numeric_column("Parch", default_value=0)
    fare_column = tf.feature_column.numeric_column("Fare", default_value=0)

    age_boundaries = get_quantile_based_boundaries(training_examples['Age'], 7)
    fare_boundaries = get_quantile_based_boundaries(training_examples['Fare'], 12)
    
    bucketized_age_column = tf.feature_column.bucketized_column(
        age_column, age_boundaries)
    bucketized_fare_column = tf.feature_column.bucketized_column(
        fare_column, fare_boundaries)
    embarked_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'Embarked', vocabulary_list=('C', 'Q', 'S'), num_oov_buckets=1)
    
                                                                       
    return set([
        pclass_column,
        sex_column,
        sibsp_column,
        parch_column,
        bucketized_age_column,
        bucketized_fare_column,
        embarked_column])


def my_training_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(500)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def my_test_input_fn(features, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(500)

    features = ds.make_one_shot_iterator().get_next()
    return features
        
    
def predict_test_input_fn(test_examples):

    _test_input_fn = my_test_input_fn(test_examples, num_epochs=1, shuffle=False)
    return _test_input_fn


def train_linear_model(
        learning_rate,
        steps,
        batch_size,
        l1_regularization_strength,
        l2_regularization_strength,
        feature_columns,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):

    periods = 10
    steps_per_period = steps / periods

    # Create a classifier object.
    my_optimizer = tf.train.FtrlOptimizer(
        learning_rate=learning_rate,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer)

    training_input_fn = lambda: my_training_input_fn(training_examples,
                                            training_targets["Survived"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_training_input_fn(training_examples,
                                                    training_targets["Survived"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_training_input_fn(validation_examples,
                                                      validation_targets["Survived"],
                                                      num_epochs=1,
                                                      shuffle=False)

    print("Training model...")
    print("LogLoss (on training data):")
    training_loglosses = []
    validation_loglosses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            # hooks=hooks,
            steps=steps_per_period)
        
        # compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        # Compute training and validation loss.
        training_logloss = metrics.log_loss(training_targets, training_probabilities)
        validation_logloss = metrics.log_loss(validation_targets, validation_probabilities)
                                  
        print("  period {0:2d} : {1:.2f}".format(period, training_logloss))
        
        training_loglosses.append(training_logloss)
        validation_loglosses.append(validation_logloss)

    print ("Model training finished.")
    return linear_classifier, training_loglosses, validation_loglosses


training_data_set = pd.read_csv("titanic_data/train.csv", sep=',')
training_data_set['Age'] = training_data_set['Age'].fillna(0)
training_data_set['Embarked'] = training_data_set['Embarked'].fillna('')

training_data_set = training_data_set.reindex(
    np.random.permutation(training_data_set.index))

training_examples = preprocess_features(training_data_set.head(500))
training_targets = preprocess_targets(training_data_set.head(500))

validation_examples = preprocess_features(training_data_set.tail(400))
validation_targets = preprocess_targets(training_data_set.tail(400))

display.display(training_examples.describe())
display.display(validation_examples.describe())
display.display(training_targets.describe())
display.display(validation_targets.describe())

corr_frame = training_examples.copy()
corr_frame['Survived'] = training_targets['Survived']
display.display(corr_frame.corr())

linear_classifier, train_losses, validation_losses = train_linear_model(
    learning_rate = 0.05,
    steps = 1000,
    batch_size = 10,
    l1_regularization_strength=0.0000,
    l2_regularization_strength=0.05,
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)


# Output a graph of loss metrics over periods.
plt.ylabel("LogLosses")
plt.xlabel("Periods")
plt.title("LogLoss vs. Periods")
plt.tight_layout()
plt.plot(train_losses, label="training")
plt.plot(validation_losses, label="validation")
plt.legend()
plt.draw()
plt.pause(10)
input("Hit Enter")


test_data_set = pd.read_csv("titanic_data/test.csv", sep=',')
test_data_set['Age'] = test_data_set['Age'].fillna(0)
test_data_set['Embarked'] = test_data_set['Embarked'].fillna('')
test_examples = preprocess_features(test_data_set)





