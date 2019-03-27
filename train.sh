#!/bin/sh

pip install xgboost
pip install numpy
pip install pandas
pip install xgboost
pip install -U scikit-learn

python ./src/train.py
