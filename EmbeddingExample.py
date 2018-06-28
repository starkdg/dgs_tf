#! /usr/bin/env python3


from __future__ import print_function
import collections
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)

train_data_uri = "embedding_data/train.tfrecord"
test_data_uri = "embedding_data/test.tfrecord"

train_path = "embedding_data/train.tfrecord"
test_path = "embedding_data/test.tfrecord"

terms_file = "embedding_data/terms.txt"

def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file
    Returns:
      A `tuple` `(labels, features)`:
      features: A dict of tensors representing the features
      labels: A tensor with the corresponding labels.
    """

    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }

    parsed_features = tf.parse_single_example(record, features)
    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels


# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def _input_fn(input_filenames, num_epochs=None, shuffle=True):

    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(25, ds.output_shapes)
    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def eval_classifier(classifier, train_path, test_path, steps):
    try:
        print("Train Model ...")
        classifier.train(
            input_fn=lambda: _input_fn([train_path]),
            steps=steps)

        evaluation_metrics = classifier.evaluate(
            input_fn=lambda: _input_fn([train_path]),
            steps=steps)

        print("Training set metrics:")
        for m in evaluation_metrics:
            print(m, evaluation_metrics[m])

        evaluation_metrics = classifier.evaluate(
            input_fn=lambda: _input_fn([test_path]),
            steps=steps)
    
        print("-"*10)
        print("Test set metrics:")
        for m in evaluation_metrics:
            print(m, evaluation_metrics[m])
        print("*"*10)
    except ValueError as err:
        print(err)


# 50 informative terms that compose our model vocabulary
informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family")

terms_column = tf.feature_column.categorical_column_with_vocabulary_file(
    key="terms", vocabulary_file=terms_file)


learning_rate = 0.01
steps = 1000
print("learning rate: ", learning_rate)
print("*"*10)


my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
feature_columns = [terms_column]
hidden_units = [50, 50]

''''
print("Linear Model Classifier")

linear_classifier = tf.estimator.LinearClassifier(
  feature_columns=feature_columns,
  optimizer=my_optimizer,
)


print("Evaluate Linear Classifier Model")
eval_classifier(linear_classifier, train_path, test_path, steps)
'''

''''
print("Evaluate DNN Classifier Model")

dnn_classifier = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.indicator_column(terms_column)],
    hidden_units=hidden_units,
    optimizer=my_optimizer)

eval_classifier(dnn_classifier, train_path, test_path, steps)
'''

print("Evaluate DNN Classifier Model with Embedding Column")

dnn_classifier2 = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.embedding_column(terms_column, dimension=10)],
    hidden_units=hidden_units,
    optimizer=my_optimizer)
   
eval_classifier(dnn_classifier2, train_path, test_path, steps)
print("Done.")


'''
var_names = dnn_classifier2.get_variable_names()
print("variable names:")
for var in var_names:
    print("{0}:shape-{1}".format(var, dnn_classifier2.get_variable_value(var).shape))
'''


''''
embedding_matrix = dnn_classifier2.get_variable_value(
'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

for term_index in range(len(informative_terms)):
    # Create a one-hot encoding for our term. It has 0s everywhere, except for
    # a single 1 in the coordinate that corresponds to that term.
    term_vector = np.zeros(len(informative_terms))
    term_vector[term_index] = 1
    # We'll now project that one-hot vector into the embedding space.
    embedding_xy = np.matmul(term_vector, embedding_matrix)
    plt.text(embedding_xy[0],
             embedding_xy[1],
             informative_terms[term_index])


# Do a little setup to make sure the plot displays nicely.
plt.rcParams["figure.figsize"] = (15, 15)
plt.xlim(1.2*embedding_matrix.min(), 1.2*embedding_matrix.max())
plt.ylim(1.2*embedding_matrix.min(), 1.2*embedding_matrix.max())
plt.draw()
plt.pause(10)
input("Hit Enter")
'''
    
