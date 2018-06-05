#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(color_codes=True)


titanic_data = pd.read_csv("titanic_data/train.csv", sep=',')
titanic_data = titanic_data.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
titanic_data['Age'] = titanic_data['Age'].fillna(0)
titanic_data['Fare'] = titanic_data['Fare'].fillna(0)


print(titanic_data.corr())

g = sns.PairGrid(titanic_data)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
plt.legend()
plt.show()
plt.pause(2)
input("hit enter")


