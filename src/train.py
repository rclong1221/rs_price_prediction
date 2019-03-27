#!/usr/bin/env_python
"""
Name:       Routhana Chan Long
WSUID:      11332872
Course:     CPTS315
Instructor: Dr. Kyle Doty

--- Usage ---
python train.py {file}
"""

import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from random import seed

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 42
FILE = './data/kc_train.csv'

def save_model(gbm):
    pickle.dump(gbm, open("./models/xgb.model", "wb"))
    print("Model saved!")

def make_model(train_X, train_Y):
    gbm = xgb.XGBRegressor(
        min_child_weight=9,
        max_depth=1,
        objective='reg:linear',
        learning_rate=0.01,
        seed=SEED,
        n_estimators=2750,
        verbose=True
    )
    print("Training...")
    gbm.fit(train_X, train_Y)
    print("Training complete!")
    return gbm

def main(f):
    '''
    Main driver - takes in k
    '''
    print('Proceeding with param k=\"%s\"' % (str(f)))

    data = pd.read_csv(f)
    data = data.drop(['id', 'date'], axis=1)
    train_X, test_X, train_Y, test_Y = train_test_split(
        data, data['price'],
        test_size=0,
        random_state=SEED
    )
    train_X = train_X.drop('price', axis=1)
    gbm = make_model(train_X, train_Y)
    save_model(gbm)

if __name__ == '__main__':
    # Default file
    f = FILE
    if len(sys.argv) == 2:
        f = sys.argv[1]
    else:
        print("Invalid parameters.  python train.py {file}")
    main(f)
