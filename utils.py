from sklearn.model_selection import train_test_split
import pandas as pd

# Load Movielens 100k dataset
data = pd.read_csv('dataset/u.data', sep='\t', names=["userId","movieId","rating","timeStamp"])

# Split dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data_df = pd.DataFrame(train_data)
test_data_df = pd.DataFrame(test_data)

train_data_df.to_csv('./dataset/train_dataset',index=False)
test_data_df.to_csv('./dataset/test_dataset',index=False)
