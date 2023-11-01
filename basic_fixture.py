import numpy as np
import pandas as pd

nom_fr = {"Marital status": "État civil",
          "Application mode": "Mode d'application",
          "Application order": "Ordre d'application",
          "Course": "Cours",
          "Daytime/evening attendance": "Présence jour/soir",
          "Previous qualification": "Qualification antérieure",
          "Nacionality": "Nationalité",
          "Mother's qualification": "Qualification mère",
          "Father's qualification": "Qualification père",
          "Mother's occupation": "Occupation mère",
          "Father's occupation": "Occupation père",
          "Displaced": "Déplacé",
          "Educational special needs": "Besoins éducatifs spéciaux",
          "Debtor": "Dettes",
          "Tuition fees up to date": "Frais de scolarité à jour",
          "Gender": "Sexe",
          "Scholarship holder": "Bourse",
          "Age at enrollment": "Âge à l'inscription",
          "International": "International",
          "Curricular units 1st sem (credited)": "Unités curriculaires 1er semestre (créditées)",
          "Curricular units 1st sem (enrolled)": "Unités curriculaires 1er semestre (inscrits)",
          "Curricular units 1st sem (evaluations)": "Unités curriculaires 1er semestre (évaluations)",
          "Curricular units 1st sem (approved)": "Unités curriculaires 1er semestre (approuvées)",
          "Curricular units 1st sem (grade)": "Unités curriculaires 1er semestre (note)",
          "Curricular units 1st sem (without evaluations)": "Unités curriculaires 1er semestre (sans évaluations)",
          "Curricular units 2nd sem (credited)": "Unités curriculaires 2e semestre (créditées)",
          "Curricular units 2nd sem (enrolled)": "Unités curriculaires 2e semestre (inscrits)",
          "Curricular units 2nd sem (evaluations)": "Unités curriculaires 2e semestre (évaluations)",
          "Curricular units 2nd sem (approved)": "Unités curriculaires 2e semestre (approuvées)",
          "Curricular units 2nd sem (grade)": "Unités curriculaires 2e semestre (note)",
          "Curricular units 2nd sem (without evaluations)": "Unités curriculaires 2e semestre (sans évaluations)",
          "Unemployment rate": "Taux de chômage",
          "Inflation rate": "Taux d'inflation",
          "GDP": "PIB",
          "Target": "Cible"}


class Dataset(object):
    _instance = None
    _raw_data = None
    target = None
    data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Dataset, cls).__new__(cls)
            cls._raw_data = pd.read_csv("./dataset.csv")

            cls.data = cls._raw_data.rename(columns=nom_fr)
            cls.data["Cible"] = cls.data["Cible"].replace("Dropout", 0)
            cls.data["Cible"] = cls.data["Cible"].replace("Graduate", 1)
            cls.data["Cible"] = cls.data["Cible"].replace("Enrolled", 2)

            # Retirer toutes les données où la cible est "Inscrit" (2)
            cls.data = cls.data.drop(cls.data[cls.data['Cible'] == 2].index)

            cls.target = cls.data["Cible"]

            # remove the target column from data
            cls.data = cls.data.drop(cls.data.columns[-1], axis=1)

            # remove other data that i simply didn't want in my stuff lolz
            # cls.data = cls.data.drop(
            #    "Unités curriculaires 2e semestre (approuvées)", axis=1)
            # cls.data = cls.data.drop(
            #    "Unités curriculaires 2e semestre (note)", axis=1)
            # cls.data = cls.data.drop(
            #    "Unités curriculaires 1er semestre (note)", axis=1)
            # cls.data = cls.data.drop(
            #    "Unités curriculaires 1er semestre (approuvées)", axis=1)
            # cls.data = cls.data.drop(
            #    "Unités curriculaires 2e semestre (inscrits)", axis=1)
            # cls.data = cls.data.drop(
            #    "Unités curriculaires 1er semestre (inscrits)", axis=1)
            # cls.data = cls.data.drop(
            #    "Unités curriculaires 2e semestre (évaluations)", axis=1)
        return cls._instance

    def get_data(self):
        return (self.data, self.target)
