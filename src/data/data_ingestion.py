import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

final_df = df[df['sentiment'].isin(['happiness','sadness'])]

final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)

train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

os.makedirs('data/raw',exist_ok=True)

train_data.to_csv('data/raw/train.csv', index=False)

test_data.to_csv('data/raw/test.csv', index=False)