import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('titanic.csv', index_col=False)
scaler = preprocessing.StandardScaler()

# Escalonamento
fare = dataframe['Fare'].values
fare_scaled = scaler.fit_transform(fare.reshape(-1, 1))
dataframe['Fare_StandardScaler'] = fare_scaled

# Normalização
scaler = preprocessing.MinMaxScaler()
fare_scaled = scaler.fit_transform(fare.reshape(-1, 1))
dataframe['Fare_MinMaxScaler'] = fare_scaled

# Ordinal encoding
encoder = preprocessing.OrdinalEncoder()
sex = dataframe['Sex'].values
sex_enconder = encoder.fit_transform(sex.reshape(-1, 1))
dataframe['Sex_OrdinalEncoder'] = sex_enconder

# One hot encoding
one_hot_encoder = preprocessing.OneHotEncoder()
sex_one_hot_encoder = one_hot_encoder.fit_transform(dataframe['Sex'].values.reshape(-1, 1))
sex_one_hot_encoder_values = sex_one_hot_encoder.toarray()
dataframe[one_hot_encoder.get_feature_names_out(['Sex'])] = sex_one_hot_encoder_values

# Tratamento de dados nulos
dataframe['Embarked'].fillna('S', inplace=True)
dataframe['Age'].fillna(dataframe['Age'].mean(), inplace=True)
dataframe['Cabin'].fillna('-', inplace=True)

# Outliers
high_filter = dataframe['Fare'].quantile(0.99)
low_filter = dataframe['Fare'].quantile(0.1)
dataframe = dataframe[(dataframe['Fare'] > low_filter) & (dataframe['Fare'] < high_filter)]
# sns.boxplot(x = dataframe['Fare'])
# plt.show() 

# Holdout 80% de dados para treinamento e 20% de dados para testes de performance
x = dataframe[['Age', 'Fare_MinMaxScaler', 'Sex_OrdinalEncoder']] # Variáveis de entrada
y = dataframe['Survived'] # Variáveis de saída
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

print(x_test)


