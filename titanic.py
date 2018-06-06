#!/usr/bin/env python3
import argparse

import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 30
pd.options.display.float_format = '{:.4f}'.format


def preprocess_features(dataframe):
    selected_features = dataframe[
        ['Pclass',
         'Sex',
         'Age',
         'Parch',
         'SibSp',
         'Embarked']
    ]

    processed_features = selected_features.copy()
    # synthetic features
    processed_features['FamilySize'] = processed_features['Parch'] + processed_features['SibSp'] + 1
    processed_features['IsMinor'] = (processed_features['Age'] < 16).astype(float)
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
    sex_col = tf.feature_column.categorical_column_with_vocabulary_list(
        'Sex', vocabulary_list=('male', 'female'), num_oov_buckets=1)
    pclass_col = tf.feature_column.categorical_column_with_identity(
        'Pclass', num_buckets=4, default_value=0)
    sex_x_pclass_col = tf.feature_column.crossed_column([sex_col, pclass_col], 12)

    return {sex_col, pclass_col, sex_x_pclass_col}


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

    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.batch(1).repeat(1)
    features = ds.make_one_shot_iterator().get_next()
    return features


def train_linear_model(
        learning_rate,
        steps,
        batch_size,
        l1_regularization_strength,
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
        l1_regularization_strength=l1_regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
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
    # validation_classes = []
    training_loglosses = []
    validation_loglosses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(input_fn=training_input_fn,
                                steps=steps_per_period)

        # compute predictions.
        training_predictions = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        validation_predictions = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        # validation_classes = np.array([item['class_ids'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_logloss = metrics.log_loss(
            training_targets, training_probabilities)
        validation_logloss = metrics.log_loss(
            validation_targets, validation_probabilities)

        print("  period {0:d} : {1:f}".format(period, training_logloss))

        training_loglosses.append(training_logloss)
        validation_loglosses.append(validation_logloss)

    print("Model Training finished")
    print("Evaluate Model")
    results = linear_classifier.evaluate(
        input_fn=predict_validation_input_fn,
        steps=100)

    for key in sorted(results):
        print(" {0} : {1}".format(key, results[key]))

    return linear_classifier, training_loglosses, validation_loglosses


training_data_set = pd.read_csv("titanic_data/train.csv", sep=',')
training_data_set['Age'].fillna(training_data_set['Age'].median(), inplace=True)
training_data_set['Fare'].fillna(training_data_set['Fare'].mean(), inplace=True)
training_data_set['Embarked'].fillna('', inplace=True)

training_data_set = training_data_set.reindex(
    np.random.permutation(training_data_set.index))

training_examples = preprocess_features(training_data_set.head(500))
training_targets = preprocess_targets(training_data_set.head(500))

validation_examples = preprocess_features(training_data_set.tail(400))
validation_targets = preprocess_targets(training_data_set.tail(400))

# print(training_examples.describe())
# print(validation_examples.describe())
# print(training_targets.describe())
# print(validation_targets.describe())

corr_frame = training_examples.copy()
corr_frame['Survived'] = training_targets['Survived']
#  print("Correltion matrix:")
#  print(corr_frame.corr())

parser = argparse.ArgumentParser(description='Machine Learning experiments')
parser.add_argument('-r', '--learning_rate', type=float, default=0.005, help="Learning rate")
parser.add_argument('-s', '--steps', type=int, default=120, help="Number of steps")
parser.add_argument('-b', '--batch_size', type=int, default=5, help="Batch Size")
parser.add_argument('-l', '--strength', type=float, default=0.00001, help="L1 Regularization Strength")

args = parser.parse_args()

learning_rate = args.learning_rate
steps = args.steps
batch_size = args.batch_size
l1_reg = args.strength
linear_classifier, train_losses, validation_losses = train_linear_model(
    learning_rate=learning_rate,
    steps=steps,
    batch_size=batch_size,
    l1_regularization_strength=l1_reg,
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

print("Try on unknown test set")
test_data_set = pd.read_csv("titanic_data/test.csv", sep=",")
test_examples = preprocess_features(test_data_set)


def predict_test_input_fn():
    return my_test_input_fn(test_examples)


test_predictions = linear_classifier.predict(predict_test_input_fn)
test_classes = np.array([item['class_ids'][0] for item in test_predictions])

test_result_df = pd.DataFrame()
test_result_df['PassengerId'] = test_data_set['PassengerId']
test_result_df['Survived'] = test_classes
print("save file to csv")
test_result_df.to_csv("titanic_data/linear_model_sex_x_pclass_submission.csv", sep=",", index=False)

input("Hit Enter")
