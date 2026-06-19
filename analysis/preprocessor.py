from sklearn.preprocessing import OneHotEncoder, StandardScaler
from analysis.loader import load_data, map_labels
import numpy as np

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

def build_feature_matrix(scaled_numeric, encoded_categorical):
    X = np.hstack((scaled_numeric, encoded_categorical))
    return X

def save_preprocessed():
    df=load_data()
    train_df = df[0]
    test_df = df[1]

    map_labels(train_df)
    map_labels(test_df)
    categorical_features = get_feature_types(train_df)[1]
    numeric_features = get_feature_types(train_df)[0]

    scaled_features = scale_features(train_df, test_df, numeric_features)
    encoded_categoricals = encode_categoricals(train_df, test_df, categorical_features)

    X_train = build_feature_matrix(scaled_features[1], encoded_categoricals[1])
    y_train = train_df["attack_category"].values
    
    X_test = build_feature_matrix(scaled_features[2], encoded_categoricals[2])
    y_test = test_df["attack_category"].values

    np.save('output/X_train.npy', X_train)
    np.save('output/y_train.npy', y_train)

    np.save('output/X_test.npy', X_test)
    np.save('output/y_test.npy', y_test)

    print("Preprocessing complete and saved to /output folder")