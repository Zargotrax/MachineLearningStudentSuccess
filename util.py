from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

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

        print(f"\n{desc[i]}")
        print(f'train_time: {t["train_time"]}, classification report: \n{
              t["classification_report"]}')
        print(f'performance: {t['performance']}')
