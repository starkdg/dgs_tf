#!/usr/bin/env python3
import argparse

import pandas as pd
import tensorflow as tf
import numpy as np
import sys
# import math
# from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 25
pd.options.display.float_format = '{:.4f}'.format


def preprocess_features(dataframe):
    selected_features = dataframe[
        ['Pclass',
         'Sex',
         'Age',
         'Parch',
         'SibSp',
         'Embarked',
         'PassengerId']]

    processed_features = selected_features.copy()
    processed_features['Age'].fillna(processed_features['Age'].median(), inplace=True)
    processed_features['Embarked'].fillna('', inplace=True)

    #  synthetic features 
    processed_features['FamilySize'] = processed_features['SibSp'] + processed_features['Parch'] + 1
    processed_features['IsMinor'] = (processed_features['Age'] < 16).astype(int)
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
    pclass_indicator_column = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity('Pclass', num_buckets=4, default_value=0))
    sex_indicator_column = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list('Sex',['male', 'female'], num_oov_buckets=1))

    return set([
        pclass_indicator_column,
        sex_indicator_column])


def my_training_input_fn(
        features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(500)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def my_test_input_fn(features):

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = tf.data.Dataset.from_tensor_slices((features))
    ds = ds.batch(1).repeat(1)
    features = ds.make_one_shot_iterator().get_next()
    return features


def train_dnn_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        l1_regularization,
        feature_columns,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):

    periods = 10
    steps_per_period = steps / periods

    # Create a classifier object.
    my_optimizer = tf.train.ProximalGradientDescentOptimizer(
        learning_rate=learning_rate,
        l1_regularization_strength=l1_regularization)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        n_classes=2,
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
        dnn_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period)

        # compute predictions.
        training_predictions = dnn_classifier.predict(
            input_fn=predict_training_input_fn)
        training_probabilities = np.array(
            [item['probabilities'][1] for item in training_predictions])

        validation_predictions = dnn_classifier.predict(
            input_fn=predict_validation_input_fn)
        validation_probabilities = np.array(
            [item['probabilities'][1] for item in validation_predictions])

        # Compute training and validation loss.
        training_logloss = metrics.log_loss(training_targets, training_probabilities)
        validation_logloss = metrics.log_loss(validation_targets, validation_probabilities)

        print("  period {0:d} : {1:.4f}".format(period, training_logloss))

        training_loglosses.append(training_logloss)
        validation_loglosses.append(validation_logloss)

    print("Model training finished.")

    results = dnn_classifier.evaluate(
        input_fn=predict_validation_input_fn, steps=steps)

    for key in sorted(results):
        print(" {0} : {1}".format(key, results[key]))

    return dnn_classifier, training_loglosses, validation_loglosses

parser = argparse.ArgumentParser(description='Machine Learning experiments')
parser.add_argument('-r', '--learning_rate', type=float, default=0.005, help="Learning rate")
parser.add_argument('-s', '--steps', type=int, default=120, help="Number of steps")
parser.add_argument('-b', '--batch_size', type=int, default=5, help="Batch Size")
parser.add_argument('-l', '--strength', type=float, default=0.00001, help="L1 Regularization Strength")
args = parser.parse_args()

print("parameters:")
print("alpha = ", args.learning_rate)
print("steps = ", args.steps)
print("batch size = ", args.batch_size)
print("l1 reg. strength = ", args.strength)

training_data_set = pd.read_csv("titanic_data/train.csv", sep=',')
training_data_set = training_data_set.reindex(np.random.permutation(training_data_set.index))

training_examples = preprocess_features(training_data_set.head(500))
training_targets = preprocess_targets(training_data_set.head(500))

validation_examples = preprocess_features(training_data_set.tail(391))
validation_targets = preprocess_targets(training_data_set.tail(391))

corr_frame = training_examples.copy()
corr_frame['Survived'] = training_targets['Survived']
# print("Correltion matrix:")
# print(corr_frame.corr())

hidden_units = [16, 16, 6]

dnn_classifier, train_losses, validation_losses = train_dnn_model(
    learning_rate=args.learning_rate,
    steps=args.steps,
    batch_size=args.batch_size,
    hidden_units=hidden_units,
    l1_regularization=args.strength,
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
plt.pause(2)
input("Hit Enter")


def eval_input_fn():
    return my_training_input_fn(validation_examples,
                                validation_targets['Survived'],
                                batch_size=1,
                                shuffle=True,
                                num_epochs=1)


print("Try on test set")

test_data_set = pd.read_csv("titanic_data/test.csv")
test_examples = preprocess_features(test_data_set)

def predict_test_input_fn():
    return my_test_input_fn(test_examples)

test_predictions = dnn_classifier.predict(predict_test_input_fn)
test_classes = np.array([item['class_ids'][0] for item in test_predictions])

test_results_df = pd.DataFrame()
test_results_df['PassengerId'] = test_examples['PassengerId']
test_results_df['Survived'] = test_classes

test_results_df.to_csv("titanic_data/dnn_nodel_submission.csv")

print("Done")
