import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import time

# Splits the data in a predictable manner (seded for constant comparison)
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
