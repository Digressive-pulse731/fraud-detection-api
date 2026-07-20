"""Standalone training entry point: python train_model.py

Trains the Isolation Forest on data/creditcard.csv and writes
model/model.joblib. All real logic lives in model/train.py.
"""

from model.train import main

if __name__ == "__main__":
    main()
