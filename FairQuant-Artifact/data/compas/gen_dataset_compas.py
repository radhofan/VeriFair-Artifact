# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,

filepath = 'propublica_data_for_fairml.csv'
column_names = ['Two_yr_Recidivism','Number_of_Priors','score_factor',
                'Age_Above_FourtyFive','Age_Below_TwentyFive',
                'African_American','Asian','Hispanic','Native_American','Other',
                'Female','Misdemeanor']
df = pd.read_csv(filepath, header=0, names=column_names)

age_column = [0 for _ in range(len(df))]
race_column = [0 for _ in range(len(df))]

for index, row in df.iterrows():
    # age: {<25 = 0, >= 25 = 1}
    if row['Age_Below_TwentyFive'] == 1:
        age_column[index] = 0
    elif row['Age_Above_FourtyFive'] == 1:
        age_column[index] = 1 
    else:
        age_column[index] = 1

    # race: {White = 0, Non-White = 1}
    if row['African_American'] == 1:
        race_column[index] = 1
    elif row['Asian'] == 1:
        race_column[index] = 1
    elif row['Hispanic'] == 1:
        race_column[index] = 1
    elif row['Native_American'] == 1:
        race_column[index] = 1
    elif row['Other'] == 1:
        race_column[index] = 1
    else: # White
        race_column[index] = 0

# print(age_column)
# print(race_column)

# add the new columns
df.insert(loc = 3, column = 'Age', value = age_column)
df.insert(loc = 4, column = 'Race', value = race_column)

# drop the originial columns
feat_to_drop = ['Age_Above_FourtyFive','Age_Below_TwentyFive',
                'African_American','Asian','Hispanic','Native_American','Other']
df = df.drop(feat_to_drop, axis=1)
# print(df)


# done preprocessing df, now create X and y
label_name = 'score_factor' # 'Two_yr_Recidivism' is a feature
X = df.drop(labels = [label_name], axis = 1, inplace = False)
y = df[label_name]
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
dataset_name = 'compas'
filename = '../' + dataset_name + '.npy'
seed = 42

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

# 80% train, 10% val, 10% test
with open(filename, 'wb') as f:
    np.save(f, X_train)
    np.save(f, y_train)
    np.save(f, X_val)
    np.save(f, y_val)
    np.save(f, X_test)
    np.save(f, y_test)