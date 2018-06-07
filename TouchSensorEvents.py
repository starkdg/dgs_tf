#! /usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

def process_events(frame):
    events = pd.DataFrame()
    events['event'] = frame['event'].apply(lambda x: np.array(x).flatten())
    return events
    

def process_classes(frame):
    targets = pd.DataFrame()
    targets['class'] = frame['class']
    return targets


def split_dataframe(df):
    #randomly permute examples
    src_df = df.reindex(np.random.permutation(df.index))

    training_df = src_df.iloc[:50,:]
    validation_df = src_df.iloc[50:100,:]
    testing_df = src_df.iloc[100:,:]

    return training_df, validation_df, testing_df
   


def create_training_input_fn(features, labels, batch_size):

    def _input_fn(num_epochs=None, Shuffle=True):
        idx = np.random.permutation(features.index)
        raw_features = {"event": features.reindex(idx)}
        raw_targets = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)
        if shuffle:
            ds = ds.shuffle(10000)
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        
    return _input_fn


def create_predict_input_fn(features, labels, batch_size):
    def _input_fn():
        raw_features = {"event": features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    return _input_fn


def construct_feature_columns():

    return set([tf.feature_column.numeric_column('events', shape=800)])


def train_model(learning_rate,
                steps,
                batch_size,
                hidden_units,
                training_examples,
                training_targets,
                validation_examples,
                validation_targets):

    periods = 10
    steps_per_period = steps / periods
    
    training_input_fn =create_training_input_fn(training_examples, training_targets, batch_size)
    predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)

    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=3,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
    )

    print("Training Model ...")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
    
        dnn_classifier.train(input_fn=training_input_fn ,steps=steps_per_period)

        training_predictions = list(dnn_classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'][0] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 3)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 3)

        training_logloss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_logloss = metrics.log_loss(validation_targets, validation_pred_one_hot)

        training_errors.append(training_logloss)
        validation_errors.append(validation_logloss)
        
        print("  period {0} : {1:.4f}".format(period, validation_logloss))
        

    map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    final_predictions = dnn_classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): ", accuracy)
    
    return dnn_classifier
    

parser = argparse.ArgumentParser(description='ML Touch Events')
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


src_df = pd.read_csv("touch_events.csv", sep=",")

print("division of classes")
print(src_df.groupby(['class']).count())

print("split data set")
traindf, valid_df, test_df = split_dataframe(src_df)

training_examples = process_events(traindf)
training_targets = process_classes(traindf)

valid_examples = process_events(valid_df)
valid_targets = process_classes(valid_df)

test_examples = process_events(test_df)
test_targets = process_classes(test_df)


hidden_units = [512, 512, 64]
classifier = train_model(args.learning_rate,
                         args.steps,
                         args.batch_size,
                         hidden_units,
                         training_examples,
                         training_targets,
                         valid_examples,
                         valid_targets)








