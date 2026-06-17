def get_feature_types(df):
    no_label_columns = df.drop(columns=['label', 'attack_category'])

    numeric_features_df = no_label_columns.select_dtypes(include='number')
    numeric_features = numeric_features_df.columns.tolist()

    categorical_features_df = no_label_columns.select_dtypes(include='object')
    categorical_features = categorical_features_df.columns.tolist()

    label_col = "attack_category"
    
    return (numeric_features, categorical_features ,label_col)