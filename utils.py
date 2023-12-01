from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from load_dataset import load_data
from matplotlib import rcParams

rcParams['figure.figsize'] = 12, 9

# Splits the data in a predictable manner (seeded for constant comparison)
# then return metrics and an evaluation of that classifier's performance


def evaluate_classifier(cls, data, target):
    t1 = time.time()
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=4)

    cls.fit(x_train, y_train)
    y_pred = cls.predict(x_test)

    train_time = time.time() - t1
    cls_report = classification_report(y_test, y_pred)
    performance = roc_auc_score(y_test, y_pred)

    report = {
        "train_time": train_time,
        "classification_report": cls_report,
        "performance": performance
    }

    return report


def compare_classifiers(classifiers, data, target, clf_names=None):
    metrics = []
    desc = []
    for i, clf in enumerate(classifiers):
        clf_name = clf_names[i] if clf_names else None
        cls_name = clf.__class__.__name__
        clf_params = clf.get_params()
        clf_desc = f'{clf_name}\n{cls_name} with params {clf_params}'

        metrics += [evaluate_classifier(clf, data, target)]
        desc += [clf_desc]

    for i in range(len(desc)):
        t = metrics[i]
        performance = t['performance']

        print(f"\n{desc[i]}")
        print(
            f'train_time: {t["train_time"]}, classification report: \n{t["classification_report"]}')
        print(f'performance: {performance}')


def heatmap(donnee):
    # Calculer la matrice de corrélation
    correlation_matrix = donnee.corr()

    # Créer un Heatmap
    plt.figure(figsize=(25, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap Correlation')
    plt.show()


def conf_matrix(true, preds):
    matrice_confusion = confusion_matrix(y_true=true, y_pred=preds)
    # Créer un Heatmap
    plt.figure(figsize=(25, 10))
    sns.heatmap(matrice_confusion, annot=True, cmap='Blues', fmt="d", cbar=False, xticklabels=[
                'Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
    plt.xlabel('Prédiction')
    plt.ylabel('Vrai')
    plt.title('Matrice de confusion')
    plt.show()


def features_importance(clf, X_train_columns, plot=False, _coef=False, log=False):
    # log=True will save this model and it's feature importances in the compile file
    if plot == False:
        if _coef:
            feat_imp = clf.coef_.T
        else:
            feat_imp = clf.feature_importances_
        feat_df = pd.DataFrame(
            feat_imp, index=X_train_columns, columns=["Importance"])
        sorted = feat_df.sort_values(by="Importance", ascending=False)

        # Saving these importances to the compilation file
        if log == True:
            s = sorted.reset_index()
            line = []
            for index, row in s.iterrows():
                ind = row['index']
                imp = row['Importance']
                line.append([ind, imp])

            add_model_feature_importances(clf, line)

        return sorted
    else:
        importance = clf.feature_importances_
        indices = np.argsort(importance)
        features = X_train_columns
        plt.title(f'Feature Importances for {clf.__class__.__name__}')
        plt.barh(range(len(indices)),
                 importance[indices], color='g', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()


def get_mean_impact_shap_category(shap_values, X_train, category):
    index_cat = X_train.columns.tolist().index(category)
    values = shap_values.values[:, index_cat]
    data = shap_values.data[:, index_cat]

    mean_shap_values = []
    value_names = []

    with open('./donnee_info_clean.json', 'r') as openfile:
        donnee_info = json.load(openfile)

    for cat_val in np.unique(data):
        mean_shap_values.append(values[np.where(data == cat_val)].mean())
        value_names.append(donnee_info[category][str(int(cat_val))])

    print(
        f"Moyenne des valeurs SHAP pour chaque catégories du feature : {category}")
    print("---------------------------------------------------------------------------")

    df = pd.DataFrame(np.array(mean_shap_values), index=np.unique(
        data), columns=["Valeur SHAP moyenne"])
    df = pd.concat([df, pd.DataFrame(value_names, index=np.unique(
        data), columns=["Nom catégorie"])], axis=1)
    return df.sort_values(by='Valeur SHAP moyenne', ascending=False)


def one_hot_shap_values(X, shap_values):
    X_cat, _, cat_features = load_data().get_data_X_y(data="simplify", OneHot=False)
    X_cat = X_cat.iloc[X.index].reset_index(drop=True)

    n_categories = []
    for feat in cat_features:
        n = X_cat[feat].nunique()
        n_categories.append(n)

    features_index = {}

    for feat in cat_features:
        if n_categories[cat_features.index(feat)] > 2:
            features_index[feat] = X.columns.to_list().index(f"{feat}_1")
        else:
            features_index[feat] = X.columns.to_list().index(f"{feat}_0")

    features_shap_values = {}

    for i in range(len(X.columns[:features_index[cat_features[0]]])):
        features_shap_values[X.columns[i]] = shap_values.values[:, i]

    for i in range(len(cat_features) - 1):
        shap_cat = np.sum(
            shap_values.values[:, features_index[cat_features[i]]:features_index[cat_features[i + 1]]], axis=1)
        features_shap_values[cat_features[i]] = shap_cat
    shap_cat = np.sum(shap_values.values[:, features_index[cat_features[len(
        cat_features) - 1]]:shap_values.values.shape[1]], axis=1)
    features_shap_values[cat_features[len(cat_features) - 1]] = shap_cat

    new_shap_values = []

    for feat in X_cat:
        new_shap_values.append(features_shap_values[feat])

    new_shap_values = np.array(new_shap_values).T

    shap_values.values = new_shap_values
    shap_values.data = X_cat.to_numpy()
    shap_values.feature_names = list(X_cat.columns)

    return shap_values, X_cat


def add_model_feature_importances(clf, features):
    # features : array of type [feature_name, importance(weight)]
    model_name = clf.__class__.__name__
    model_params = clf.get_params()

    line = [model_name, model_params]
    for feat in features:
        ind, imp = feat
        line.append(ind)
        line.append(imp)

    # Check if the data array has the correct number of elements
    if len(line) != 272:
        to_adjust = int((272 - len(line)) / 2)
        for i in range(to_adjust):
            line.append("None")
            line.append(0)

    # Read the existing CSV file or create an empty DataFrame if the file doesn't exist
    header = None
    try:
        df_existing = pd.read_csv("./compil/features_importance.csv")
        header = df_existing.columns.tolist()
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=header)

    df = pd.DataFrame(columns=header)

    # Extract model_name, params, and feature-weight pairs from the line
    model_name, params, *feature_weight_pairs = line

    # Organize feature-weight pairs into dictionaries
    features = {
        f"feature_{i}": feature_weight_pairs[i-1] for i in range(1, 135)}
    weights = {
        f"feature_{i}_weight": feature_weight_pairs[i] for i in range(135, 269)}

    # Create a new row with the received data
    new_row = pd.DataFrame(
        [[model_name, params] + list(features.values()) + list(weights.values())], columns=header)

    # Concatenate the existing DataFrame with the new row
    df = pd.concat([df_existing, new_row], ignore_index=True)

    # Save the DataFrame to the CSV file and close the file
    df.to_csv("./compil/features_importance.csv", index=False)
    print("File features_importances.csv has been updated and saved.")
