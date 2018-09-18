import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

fights = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/fights.csv"))
fighters = pd.read_csv(os.path.join(os.path.dirname(__file__),"data/fighters.csv"))

# fighter stat labels
stats = ["SLPM", "SAPM", "STRA", "STRD", "TD", "TDA", "TDD", "SUBA"]

# handles duplicates in fighters
fighters.iloc[1462, 1] = 'Dong Hyun Kim 170'
fighters.iloc[1925, 1] = 'Michael McDonald 205'

# prints out duplicates
dups = fighters[fighters.duplicated(subset="NAME", keep=False)]
for dup in dups.iterrows():
    print (dup)

# convert percentage entries into decimals
for col in ["STRA", "STRD", "TDA", "TDD"]:
	fighters[col] = list(map(lambda x: x.strip('%'), fighters[col]))
	fighters[col] = list(map(lambda x: x / 100, fighters[col].astype(np.float32)))

# cleans dataset by removing fighters without stats
fighters = fighters.loc[~((fighters[stats[0]] == 0) & (fighters[stats[1]] == 0) & (fighters[stats[2]] == 0) & (fighters[stats[3]] == 0) & (fighters[stats[4]] == 0) & (fighters[stats[5]] == 0) & (fighters[stats[6]] == 0) & (fighters[stats[7]] == 0))]

# writes out the cleansed fighter data to an outfile
fighters.to_csv(os.path.join(os.path.dirname(__file__), "data/fighters_clean.csv"))

# keeps only the relevant information i.e. stats and the names
fighters = fighters.loc[:, ["NAME"] + stats]

#sets name as the index
fighters.set_index("NAME", inplace = True)

# converts the types of the stats as numpy float32
fighters.loc[:, stats] = fighters.loc[:, stats].astype(np.float32)

# handles duplicates in fights
for col in ["Fighter1", "Fighter2", "Winner"]:
    fights[col][(fights[col] == "Michael McDonald") &
                (fights["WeightClass"] == "Light Heavyweight")] = 'Michael McDonald 205'
    fights[col][(fights[col] == "Dong Hyun Kim") &
                (fights["WeightClass"] == "Welterweight")] = 'Dong Hyun Kim 170'

# keeps only relevant information i.e. player names and outcome
fights = fights.loc[:, ["Fighter1", "Fighter2", "Winner"]]

# by default Fighter1 is the Winner, so lets randomly swap Fighter1 and Fighter2 for about half of the indices
swapped_index = np.random.choice(len(fights), size = int(len(fights)/2), replace = False)
fights.iloc[swapped_index, [0, 1]] = fights.iloc[swapped_index, [1, 0]].values

print(fights.head(10))

# converts outcome to a binary result
fights["outcome"] = (fights["Fighter1"] == fights["Winner"]).astype("int")
fights.drop("Winner", axis = 1, inplace = True)

# removes fight entries where we don't have fighter data
available_fighter_names = fighters.index.values.tolist()
fights = fights.loc[(fights["Fighter1"].isin(available_fighter_names)) & (fights["Fighter2"].isin(available_fighter_names))]

# adds columns each of which indicate the diff of each stat between Fighter1 and Fighter2
for col in stats:
	fights[col] = fights.apply(lambda row: fighters.loc[row["Fighter1"], col] - fighters.loc[row["Fighter2"], col], axis = 1)

# writes out the cleansed fight data to an outfile
fights.to_csv(os.path.join(os.path.dirname(__file__), "data/fights_clean.csv"))

fights.drop(["Fighter1", "Fighter2"], axis = 1, inplace = True)

# prepare the x, y datasets
X, y = fights.iloc[:, 1:], fights.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("Datasets have been prepared")

# normalizes the data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
print("Datasets have been normalized")

model = Sequential()

model.add(Dense(16, input_dim=X_train_norm.shape[1],
                activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x=X_train_norm, y=y_train, epochs=200, batch_size=64, verbose=0)
test_results = model.evaluate(x = X_test_norm, y = y_test, verbose=0)
print("Test Accuracy = {}".format(test_results[1]))

model.save(os.path.join(os.path.dirname(__file__), "cache.h5"))

'''# intializes a linear support vector machine
ridge_reg = linear_model.Ridge(alpha = 0.5)
print("Ridge Regressor has been initialized")

# train the svr over the datasets
ridge_reg.fit(X_train_norm, y_train)
print("Ridge Regressor have been trained")

# predict and test
svr_lin_pred = ridge_reg.predict(X_test_norm)
print("Results have been predicted")

print("Ridge Regressor accuracy: ",  accuracy_score(y_test, ridge_reg.predict(X_test_norm).round()))

print(ridge_reg)
#Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
# normalize=False, random_state=None, solver='auto', tol=0.001)'''

