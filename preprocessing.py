from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    categorical_cols = ['protocol_type', 'service', 'flag']

    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # encode label
    data['label'] = le.fit_transform(data['label'])

    return data