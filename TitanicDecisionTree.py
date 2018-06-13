#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn import metrics, tree
import graphviz

pd.options.display.max_rows = 30
pd.options.display.float_format = '{:.4f}'.format


def preprocess_features(df):
    selected_features = df[
        ['Pclass',
         'Sex',
         'Age',
         'Parch',
         'SibSp',
         'Fare',
         'Embarked']
    ]

    processed_features = selected_features.copy()
    processed_features['Age'].fillna(-1, inplace=True)
    processed_features['Embarked'].fillna('None', inplace=True)
    processed_features['Fare'].fillna(1000.0, inplace=True)

    sex_cat_type = CategoricalDtype(categories=['male', 'female', 'none'], ordered=False)
    processed_features['Sex'] = processed_features['Sex'].astype(
        sex_cat_type).cat.codes.astype(int)

    embarked_cat_type = CategoricalDtype(categories=['S', 'C', 'Q', 'None'], ordered=False)
    processed_features['Embarked'] = processed_features['Embarked'].astype(
        embarked_cat_type).cat.codes.astype(int)

    processed_features['FamilySize'] = processed_features['Parch'] + processed_features['SibSp'] + 1
    processed_features['IsMinor'] = (processed_features['Age'] < 16).astype(int)
    
    return processed_features


def preprocess_targets(dataframe):
    output_targets = pd.DataFrame()
    output_targets['Survived'] = dataframe["Survived"].astype(int)
    return output_targets


def train_decision_tree(examples, targets):
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(examples, targets)
    return classifier


def run_tests(classifier, examples, targets):
    classes = classifier.predict(examples)
    target_names = ['Non-Survived', 'Survived']
    print("Decision tree accuracy: ", metrics.accuracy_score(targets, classes))
    print(metrics.classification_report(
        targets, classes, target_names=target_names))


training_data_set = pd.read_csv("titanic_data/train.csv", sep=',')
training_data_set = training_data_set.reindex(np.random.permutation(training_data_set.index))

training_examples = preprocess_features(training_data_set.head(500))
training_targets = preprocess_targets(training_data_set.head(500))

validation_examples = preprocess_features(training_data_set.tail(400))
validation_targets = preprocess_targets(training_data_set.tail(400))

classifier = train_decision_tree(training_examples, training_targets)
run_tests(classifier, validation_examples, validation_targets)

feature_names = ['Pclass', 'Sex', 'Age', 'Parch',
                 'SibSp', 'Fare', 'Embarked', 'FamilySize', 'IsMinor']

print("export graph to titanic.pdf")
dot_data = tree.export_graphviz(classifier,
                                out_file=None,
                                class_names=['N', 'S'],
                                feature_names=feature_names,
                                filled=True,
                                rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("titanic")

print("Try on test data")
test_data_set = pd.read_csv("titanic_data/test.csv", sep=",")

test_examples = preprocess_features(test_data_set)
predicted_classes = classifier.predict(test_examples)

test_results = pd.DataFrame()
test_results['PassengerId'] = test_data_set['PassengerId']
test_results['Survived'] = pd.Series(predicted_classes)
results_file_name = "titanic_decision_tree.csv"
print("save to ", results_file_name)
test_results.to_csv(results_file_name, index=False)
print("Done.")



