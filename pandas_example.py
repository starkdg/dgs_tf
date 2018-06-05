#! /usr/bin/env python3

import pandas as pd
import numpy as np

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({'City Name': city_names, "Population": population})


print(type(cities['City Name']))
print(cities['City Name'])

print(type(cities['City Name'][1]))
print(cities['City Name'][1])

print(type(cities[0:2]))
print(cities[0:2])


1
cities['Area'] = pd.Series([46.87, 176.53, 97.92])
cities['Density'] = cities['Population'] / cities['Area']

select1 = cities['Area'].apply(lambda val: val > 50)
select2 = city_names.apply(lambda val: val.startswith("San"))
cities["Selection"] = select1 & select2


# randomly permute rows
print(cities)
cities = cities.reindex(np.random.permutation(cities.index))
print(cities)

cities = cities.reindex(np.random.permutation(cities.index))
print(cities)

cities = cities.reindex(np.random.permutation(cities.index))
print(cities)


# allow for index ranges outside of normal range
cities = cities.reindex([0, 2, 4])
print(cities)

cities = cities.reindex([5, 6, 7])
print(cities)


cities = cities.reindex([0, 4, 5, 2])
print(cities)

urlstr = "https://storage.googleapis.com"
filestr = "mledu-datasets/california_housing_train.csv"

resourcestr = urlstr + "/" + filestr

calif_housing_dataframe = pd.read_csv(resourcestr, sep=",")

print(calif_housing_dataframe.describe())

print(calif_housing_dataframe.head())
