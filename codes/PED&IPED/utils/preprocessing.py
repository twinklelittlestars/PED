import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def extract_input_features(dataset_scheme):
    input_features = {}
    for attr in dataset_scheme:
        input_features[attr] = [feat for feat in dataset_scheme if feat != attr]
    return input_features


def train_joint_probability_models(data_path, input_features):
    data = pd.read_csv(data_path, dtype=str)
    models = {}

    for target_attr, features in input_features.items():
        if not features:
            print(f"Attribute {target_attr} has no input features, skipping training.")
            continue

        X = data[features]
        y = data[target_attr]

        x_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_encoded = x_enc.fit_transform(X)

        y_enc = LabelEncoder()
        y_encoded = y_enc.fit_transform(y)

        clf = MultinomialNB(fit_prior=True)
        clf.fit(X_encoded, y_encoded)

        models[target_attr] = {
            "model": clf,
            "x_encoder": x_enc,
            "y_encoder": y_enc,
        }

    return models


def predict_attribute_probabilities(data_path, models, input_features):
    data = pd.read_csv(data_path, dtype=str)
    probabilities = pd.DataFrame(index=data.index)

    for target_attr, model_info in models.items():
        features = input_features.get(target_attr, [])
        if not features:
            print(
                f"Attribute {target_attr} has no input features, skipping prediction."
            )
            probabilities[target_attr] = None
            continue

        X = data[features]
        y_actual = data[target_attr]

        x_encoder = model_info["x_encoder"]
        y_encoder = model_info["y_encoder"]
        clf = model_info["model"]

        X_encoded = x_encoder.transform(X)
        y_actual_encoded = y_encoder.transform(y_actual)
        y_pred_proba = clf.predict_proba(X_encoded)

        actual_probabilities = [
            y_pred_proba[i, y_actual_encoded[i]] for i in range(len(y_actual))
        ]
        probabilities[target_attr] = actual_probabilities

    return probabilities


def train_marginal_probability_models(data_path):
    data = pd.read_csv(data_path, dtype=str)
    models = {}

    for attr in data.columns:
        value_counts = data[attr].value_counts(normalize=True)
        models[attr] = value_counts.to_dict()

    return models


def normalize_probabilities_to_range(probabilities):
    min_prob = probabilities.min()
    max_prob = probabilities.max()
    return (probabilities - min_prob) / (max_prob - min_prob)


def predict_marginal_probabilities(data_path, models):
    data = pd.read_csv(data_path, dtype=str)
    probabilities = pd.DataFrame(index=data.index)

    for attr in data.columns:
        model = models.get(attr, {})
        probabilities[attr] = data[attr].apply(lambda x, m=model: m.get(x, 0))
        probabilities[attr] = normalize_probabilities_to_range(probabilities[attr])

    return probabilities


def generate_probability_files(dirty_file, output_prob_file, dataset_scheme):
    input_feats = extract_input_features(dataset_scheme)
    joint_models = train_joint_probability_models(dirty_file, input_feats)
    joint_probs = predict_attribute_probabilities(dirty_file, joint_models, input_feats)
    joint_probs.to_csv(output_prob_file, index=False)
    print(f"Joint probabilities saved to: {output_prob_file}")

    # -- If you also want marginal:
    # marginal_models = train_marginal_probability_models(dirty_file)
    # marginal_probs = predict_marginal_probabilities(dirty_file, marginal_models)
    # marginal_out = output_prob_file.replace('.csv', '_marginal.csv')
    # marginal_probs.to_csv(marginal_out, index=False)
    # print(f"Marginal probabilities saved to: {marginal_out}")
