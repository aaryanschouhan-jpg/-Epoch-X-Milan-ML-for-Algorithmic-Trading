import pandas as pd
from src.model import Model

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    model = Model()
    model.train(df)
    print('Model training complete.')
