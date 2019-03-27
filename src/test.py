#!/usr/bin/env_python
"""
Name:       Routhana Chan Long
WSUID:      11332872
Course:     CPTS315
Instructor: Dr. Kyle Doty

--- Usage ---
python tests.py {file} {model}
"""

import sys
import numpy as np
import pandas as pd
import pickle
import time

from random import seed

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 42
FILE = './data/kc_test.csv'
MODEL = './models/xgb.model'

def save_stats(data):
    s = ""
    for i in range(len(data['errors'])):
        s += ("Actual: %s  Predicted: %s  Error: %s  Absolute Error: %s\n"
            % (str(data['test_Y'][i]), str(data['pred_Y'][i]), str(data['errors'][i]), str(data['absolute_errors'][i])))
    s += ("MAE: %f\n" % data['mae'])
    s += ("Prediction Time: %f\n" % data['prediction_time'])
    print("Writing to output.txt...")
    with open('./output.txt', 'w') as file:
        file.write(s)
    print("Writing complete...")
    print(s)

def load_model(f):
    gbm = pickle.load(open(f, "rb"))
    return gbm

def main(f, m):
    '''
    Main driver - takes in k
    '''
    print('Proceeding with param k=\"%s\"' % (str(f)))

    outputs = dict()

    data = pd.read_csv(f)
    data = data.drop(['id', 'date'], axis=1)
    outputs['test_Y'] = data['price'].tolist()
    test_X = data.drop('price', axis=1)

    gbm = load_model(m)

    start = time.time()
    outputs['pred_Y'] = gbm.predict(test_X)
    end = time.time()

    outputs['prediction_time'] = (end - start) / len(outputs['pred_Y'])
    outputs['errors'] = outputs['test_Y'] - outputs['pred_Y']
    outputs['absolute_errors'] = np.absolute(outputs['errors'])
    outputs['mae'] = np.median(outputs['absolute_errors'])

    save_stats(outputs)

if __name__ == '__main__':
    # Default file
    f = FILE
    m = MODEL
    if len(sys.argv) == 3:
        f = sys.argv[1]
        m = sys.argv[2]
    else:
        print("Invalid parameters.  python train.py {file} {model}")
    main(f, m)
