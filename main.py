from analysis.loader import load_data

df=load_data()
train_df = df[0]
test_df = df[1]
print(train_df.head())
