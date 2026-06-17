from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

def get_feature_types(df):
    no_label_columns = df.drop(columns=['label', 'attack_category'])

    numeric_features_df = no_label_columns.select_dtypes(include='number')
    numeric_features = numeric_features_df.columns.tolist()

    categorical_features_df = no_label_columns.select_dtypes(include='object')
    categorical_features = categorical_features_df.columns.tolist()

    label_col = "attack_category"

    return (numeric_features, categorical_features ,label_col)

def encode_categoricals(train_df, test_df, categorical_features):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_train = encoder.fit_transform(train_df[categorical_features])
    encoded_test = encoder.transform(test_df[categorical_features])
    return (encoder, encoded_train, encoded_test)

def scale_features(train_df, test_df, numeric_features):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_df[numeric_features])
    scaled_test = scaler.transform(test_df[numeric_features])
    return (scaler, scaled_train, scaled_test)