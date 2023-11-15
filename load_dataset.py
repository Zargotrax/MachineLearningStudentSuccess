import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#-------------------------------------------------------------------
# Ce fichier contient la classe pour instancier notre jeu de données
#-------------------------------------------------------------------


class load_data():
    def __init__(self):
        # On va instancier le dataset
        data = pd.read_csv("./dataset.csv")

        # Convertir les noms de caractéristiques en français
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

        data = data.rename(columns=nom_fr)
        data["Cible"] = data["Cible"].replace("Dropout", 0)
        data["Cible"] = data["Cible"].replace("Graduate", 1)
        data["Cible"] = data["Cible"].replace("Enrolled", 2)
        # Retirer toutes les données où la cible est "Inscrit" (2)
        data = data.drop(data[data['Cible'] == 2].index).reset_index(drop=True)
        
        self.cat_feat = ["État civil", "Mode d'application", "Cours", "Présence jour/soir", "Qualification antérieure",
                        "Nationalité", "Qualification mère", "Qualification père", "Occupation mère", "Occupation père",
                        "Déplacé", "Besoins éducatifs spéciaux", "Dettes", "Frais de scolarité à jour", "Sexe", "Bourse",
                        "International"]
        self.num_feat = ["Ordre d'application", "Âge à l'inscription", "Unités curriculaires 1er semestre (créditées)",
                  "Unités curriculaires 1er semestre (inscrits)", "Unités curriculaires 1er semestre (évaluations)",
                  "Unités curriculaires 2e semestre (créditées)", "Unités curriculaires 2e semestre (inscrits)",
                  "Unités curriculaires 2e semestre (évaluations)", "Taux de chômage", "Taux d'inflation", "PIB"]
        self.data = data


    def get_original_dataframe(self, keep_type=False):
        df = self.data.copy() # Prendre en copie le jeux de données
        if keep_type == False:
            # On va changer les types pour les données catégoriques afin que les modèles puisse comprendre
            for feature in self.cat_feat:
                df[feature] = df[feature].astype('category')

        return df, list(df.select_dtypes(['category']).columns)
    
    def get_simplify_dataframe(self):
        # TODO WORK IN PROGRESS

        df = self.data.copy() # Prendre en copie le jeux de données
        
        # Simplifier Mode d'application, Cours, Qualification antérieure, Qualification père/mère, Occupation père/mère

        # Mode d'application [1]
        # 1 — 1ère phase (1, 3, 7)
        index_1ere_phase = self.data.index[(self.data["Mode d'application"] == 1) | (self.data["Mode d'application"] == 3) | (self.data["Mode d'application"] == 7)].tolist()
        df.iloc[index_1ere_phase, 1] = 1
        # 2 - 2e phase (8)
        index_2e_phase = self.data.index[(self.data["Mode d'application"] == 8)].tolist()
        df.iloc[index_2e_phase, 1] = 2
        # 3 - 3e phase (9)
        index_3e_phase = self.data.index[(self.data["Mode d'application"] == 9)].tolist()
        df.iloc[index_3e_phase, 1] = 3
        # 4 - Titulaire diplôme (Degree, Bachelor, Master, Doctorate) (4)
        index_titulaire_diplome = self.data.index[(self.data["Mode d'application"] == 4)].tolist()
        df.iloc[index_titulaire_diplome, 1] = 4
        # 5 - Diplôme spécialisation/technologique (15)
        index_specialisation_diplome = self.data.index[(self.data["Mode d'application"] == 15)].tolist()
        df.iloc[index_specialisation_diplome, 1] = 5       
        # 6 - Diplôme cycle court (17)
        index_cycle_court_diplome = self.data.index[(self.data["Mode d'application"] == 17)].tolist()
        df.iloc[index_cycle_court_diplome, 1] = 6        
        # 7 - Transfert (13)
        index_transfert = self.data.index[(self.data["Mode d'application"] == 13)].tolist()
        df.iloc[index_transfert, 1] = 7      
        # 8 - Changement d'établissement/cours (14, 16, 18)
        index_changement = self.data.index[(self.data["Mode d'application"] == 14) | (self.data["Mode d'application"] == 16) | (self.data["Mode d'application"] == 18)].tolist()
        df.iloc[index_changement, 1] = 8        
        # 9 - Ordonnance (2, 5, 10, 11) # TODO À voir ce que ça signifie
        index_ordonnance = self.data.index[(self.data["Mode d'application"] == 2) | (self.data["Mode d'application"] == 5) | (self.data["Mode d'application"] == 10) | (self.data["Mode d'application"] == 11)].tolist()
        df.iloc[index_ordonnance, 1] = 9          
        # 10 - Étudiant international (baccalauréat) (6)
        index_international = self.data.index[(self.data["Mode d'application"] == 6)].tolist()
        df.iloc[index_international, 1] = 10       
        # 11 - Plus de 23 ans (12)
        index_plus23 = self.data.index[(self.data["Mode d'application"] == 12)].tolist()
        df.iloc[index_plus23, 1] = 11
           
        # Cours [3] 
        # Grouper le cours de Gestion et Gestion (présence soir) ainsi que Service Social et Service Social (présence soir)
        # 1 Technologies de production de biocarburants (1)
        index_cours = self.data.index[np.isin(self.data["Cours"], [1])].tolist()
        df.iloc[index_cours, 3] = 1
        # 2 Conception d'animation et multimédia (2)
        index_cours = self.data.index[np.isin(self.data["Cours"], [2])].tolist()
        df.iloc[index_cours, 3] = 2
        # 3 Agronomie (4)
        index_cours = self.data.index[np.isin(self.data["Cours"], [4])].tolist()
        df.iloc[index_cours, 3] = 3
        # 4 Conception de communications (5)
        index_cours = self.data.index[np.isin(self.data["Cours"], [5])].tolist()
        df.iloc[index_cours, 3] = 4
        # 5 Soins infirmiers vétérinaires (6)
        index_cours = self.data.index[np.isin(self.data["Cours"], [6])].tolist()
        df.iloc[index_cours, 3] = 5
        # 6 Génie informatique (7)
        index_cours = self.data.index[np.isin(self.data["Cours"], [7])].tolist()
        df.iloc[index_cours, 3] = 6
        # 7 Équiniculture (8)
        index_cours = self.data.index[np.isin(self.data["Cours"], [8])].tolist()
        df.iloc[index_cours, 3] = 7
        # 8 Gestion (9, 17)
        index_cours = self.data.index[np.isin(self.data["Cours"], [9, 17])].tolist()
        df.iloc[index_cours, 3] = 8
        # 9 Service social (10, 3)
        index_cours = self.data.index[np.isin(self.data["Cours"], [10, 3])].tolist()
        df.iloc[index_cours, 3] = 9
        # 10 Tourisme (11)
        index_cours = self.data.index[np.isin(self.data["Cours"], [11])].tolist()
        df.iloc[index_cours, 3] = 10
        # 11 Soins infirmier (12)
        index_cours = self.data.index[np.isin(self.data["Cours"], [12])].tolist()
        df.iloc[index_cours, 3] = 11
        # 12 Hygiène buccale (Dentiste ou Hygiéniste dentaire) (13)
        index_cours = self.data.index[np.isin(self.data["Cours"], [13])].tolist()
        df.iloc[index_cours, 3] = 12
        # 13 Gestion de la publicité et du marketing (14)
        index_cours = self.data.index[np.isin(self.data["Cours"], [14])].tolist()
        df.iloc[index_cours, 3] = 13
        # 14 Journalisme et communication (15)
        index_cours = self.data.index[np.isin(self.data["Cours"], [15])].tolist()
        df.iloc[index_cours, 3] = 14
        # 15 Éducation de base (16)
        index_cours = self.data.index[np.isin(self.data["Cours"], [16])].tolist()
        df.iloc[index_cours, 3] = 15

        # Qualification antérieure [5]
        # 1 Étude supérieure – doctorat (5)
        index_doc = self.data.index[(self.data["Qualification antérieure"] == 5)].tolist()
        df.iloc[index_doc, 5] = 1
        # 2 Étude supérieure – maîtrise (4, 17)
        index_maitrise = self.data.index[(self.data["Qualification antérieure"] == 4) | (self.data["Qualification antérieure"] == 17)].tolist()
        df.iloc[index_maitrise, 5] = 2
        # 3 Étude supérieure – baccalauréat (2)
        index_bacc = self.data.index[(self.data["Qualification antérieure"] == 2)].tolist()
        df.iloc[index_bacc, 5] = 3
        # 4 Étude supérieure – diplôme (3, 15)
        index_diplome = self.data.index[(self.data["Qualification antérieure"] == 3) | (self.data["Qualification antérieure"] == 15)].tolist()
        df.iloc[index_diplome, 5] = 4
        # 5 Fréquence d'étude supérieure (6)
        index_freq = self.data.index[(self.data["Qualification antérieure"] == 6)].tolist()
        df.iloc[index_freq, 5] = 5
        # 6 Cours spécialisation/technique (14, 16)
        index_specialisation = self.data.index[(self.data["Qualification antérieure"] == 14) | (self.data["Qualification antérieure"] == 16)].tolist()
        df.iloc[index_specialisation, 5] = 6
        # 7 Étude secondaire – 12e année de scolarité ou équivalent (1)
        index_secondaire = self.data.index[(self.data["Qualification antérieure"] == 1)].tolist()
        df.iloc[index_secondaire, 5] = 7       
        # 8 Éducation de base (7, 8, 9, 10, 11, 12, 13)
        index_edu_base = self.data.index[(self.data["Qualification antérieure"] == 7) | (self.data["Qualification antérieure"] == 8)| (self.data["Qualification antérieure"] == 9) | (self.data["Qualification antérieure"] == 10) | (self.data["Qualification antérieure"] == 11) | (self.data["Qualification antérieure"] == 12) | (self.data["Qualification antérieure"] == 13)].tolist()
        df.iloc[index_edu_base, 5] = 8

        # Qualification père/mère [7 (mère), 8 (père)]
        # 1 Étude supérieure – doctorat (5, 34)
        index_doc_mere = self.data.index[(self.data["Qualification mère"] == 5) | (self.data["Qualification mère"] == 34)].tolist()
        df.iloc[index_doc_mere, 7] = 1
        index_doc_pere = self.data.index[(self.data["Qualification père"] == 5) | (self.data["Qualification père"] == 34)].tolist()
        df.iloc[index_doc_pere, 8] = 1
        # 2 Étude supérieure – maîtrise (4, 33)
        index_maitrise_mere = self.data.index[(self.data["Qualification mère"] == 4) | (self.data["Qualification mère"] == 33)].tolist()
        df.iloc[index_maitrise_mere, 7] = 2
        index_maitrise_pere = self.data.index[(self.data["Qualification père"] == 4) | (self.data["Qualification père"] == 33)].tolist()
        df.iloc[index_maitrise_pere, 8] = 2
        # 3 Étude supérieure – baccalauréat (2)
        index_bacc_mere = self.data.index[(self.data["Qualification mère"] == 2)].tolist()
        df.iloc[index_bacc_mere, 7] = 3
        index_bacc_pere = self.data.index[(self.data["Qualification père"] == 2)].tolist()
        df.iloc[index_bacc_pere, 8] = 3
        # 4 Étude supérieure – diplôme (3, 30)
        index_diplome_mere = self.data.index[(self.data["Qualification mère"] == 3) | (self.data["Qualification mère"] == 30)].tolist()
        df.iloc[index_diplome_mere, 7] = 4
        index_diplome_pere = self.data.index[(self.data["Qualification père"] == 3) | (self.data["Qualification père"] == 30)].tolist()
        df.iloc[index_diplome_pere, 8] = 4
        # 5 Fréquence d'étude supérieure (6)
        index_freq_mere = self.data.index[(self.data["Qualification mère"] == 6)].tolist()
        df.iloc[index_freq_mere, 7] = 5
        index_freq_pere = self.data.index[(self.data["Qualification père"] == 6)].tolist()
        df.iloc[index_freq_pere, 8] = 5
        # 6 Cours spécialisation/technique (16, 29, 31, 32)
        index_spe_mere = self.data.index[(self.data["Qualification mère"] == 16) | (self.data["Qualification mère"] == 29) | (self.data["Qualification mère"] == 31) | (self.data["Qualification mère"] == 32)].tolist()
        df.iloc[index_spe_mere, 7] = 6
        index_spe_pere = self.data.index[(self.data["Qualification père"] == 16) | (self.data["Qualification père"] == 29) | (self.data["Qualification père"] == 31) | (self.data["Qualification père"] == 32)].tolist()
        df.iloc[index_spe_pere, 8] = 6
        # 7 Cours administration/commerce (13, 22, 23)
        index_adm_mere = self.data.index[(self.data["Qualification mère"] == 13) | (self.data["Qualification mère"] == 22) | (self.data["Qualification mère"] == 23)].tolist()
        df.iloc[index_adm_mere, 7] = 7
        index_adm_pere = self.data.index[(self.data["Qualification père"] == 13) | (self.data["Qualification père"] == 22) | (self.data["Qualification père"] == 23)].tolist()
        df.iloc[index_adm_pere, 8] = 7
        # 8 Étude secondaire (1)
        index_sec_mere = self.data.index[(self.data["Qualification mère"] == 1)].tolist()
        df.iloc[index_sec_mere, 7] = 8
        index_sec_pere = self.data.index[(self.data["Qualification père"] == 1)].tolist()
        df.iloc[index_sec_pere, 8] = 8
        # 9 Cours complémentaires (11, 15)
        index_comp_mere = self.data.index[(self.data["Qualification mère"] == 11) | (self.data["Qualification mère"] == 15)].tolist()
        df.iloc[index_comp_mere, 7] = 9
        index_comp_pere = self.data.index[(self.data["Qualification père"] == 11) | (self.data["Qualification père"] == 15)].tolist()
        df.iloc[index_comp_pere, 8] = 9
        # 10 Éducation de base (7, 8, 9, 10, 12, 14, 17, 18, 19, 20, 21, 27, 28)
        index_base_mere = self.data.index[(self.data["Qualification mère"] == 7) | (self.data["Qualification mère"] == 8) | (self.data["Qualification mère"] == 9) | (self.data["Qualification mère"] == 10) | (self.data["Qualification mère"] == 12) | (self.data["Qualification mère"] == 14) | (self.data["Qualification mère"] == 17) | (self.data["Qualification mère"] == 18) | (self.data["Qualification mère"] == 19) | (self.data["Qualification mère"] == 20) | (self.data["Qualification mère"] == 21) | (self.data["Qualification mère"] == 27) | (self.data["Qualification mère"] == 28)].tolist()
        df.iloc[index_base_mere, 7] = 10
        index_base_pere = self.data.index[(self.data["Qualification père"] == 7) | (self.data["Qualification père"] == 8) | (self.data["Qualification père"] == 9) | (self.data["Qualification père"] == 10) | (self.data["Qualification père"] == 12) | (self.data["Qualification père"] == 14) | (self.data["Qualification père"] == 17) | (self.data["Qualification père"] == 18) | (self.data["Qualification père"] == 19) | (self.data["Qualification père"] == 20) | (self.data["Qualification père"] == 21) | (self.data["Qualification père"] == 27) | (self.data["Qualification père"] == 28)].tolist()
        df.iloc[index_base_pere, 8] = 10
        # 11 Aucune éducation (25, 26)
        index_auc_mere = self.data.index[(self.data["Qualification mère"] == 25) | (self.data["Qualification mère"] == 26)].tolist()
        df.iloc[index_auc_mere, 7] = 11
        index_auc_pere = self.data.index[(self.data["Qualification père"] == 25) | (self.data["Qualification père"] == 26)].tolist()
        df.iloc[index_auc_pere, 8] = 11
        # 12 Inconnu (24)
        index_inconnu_mere = self.data.index[(self.data["Qualification mère"] == 24)].tolist()
        df.iloc[index_inconnu_mere, 7] = 12
        index_inconnu_pere = self.data.index[(self.data["Qualification père"] == 24)].tolist()
        df.iloc[index_inconnu_pere, 8] = 12

        # Occupation père/mère [9 (mère), 10 (père)]
        # 1 Étudiant (1)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [1])].tolist()
        df.iloc[index_mere, 9] = 1
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [1])].tolist()
        df.iloc[index_pere, 10] = 1
        # 2 Représentants du Pouvoir Législatif et des Organes Exécutifs, Directeurs et Dirigeants Exécutifs (2)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [2])].tolist()
        df.iloc[index_mere, 9] = 2
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [2])].tolist()
        df.iloc[index_pere, 10] = 2
        # 3 Spécialistes en activités intellectuelles et scientifiques (3, 19)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [3, 19])].tolist()
        df.iloc[index_mere, 9] = 3
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [3, 19])].tolist()
        df.iloc[index_pere, 10] = 3
        # 4 Techniciens et professions de niveau intermédiaire (4, 23, 24, 25, 26)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [4, 23, 24, 25, 26])].tolist()
        df.iloc[index_mere, 9] = 4
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [4, 23, 24, 25, 26])].tolist()
        df.iloc[index_pere, 10] = 4
        # 5 Personnel administratif (5)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [5])].tolist()
        df.iloc[index_mere, 9] = 5
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [5])].tolist()
        df.iloc[index_pere, 10] = 5
        # 6 Travailleurs des services personnels, de la sécurité et de la sûreté et vendeurs (6, 30, 31, 32, 33)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [6, 30, 31, 32, 33])].tolist()
        df.iloc[index_mere, 9] = 6
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [6, 30, 31, 32, 33])].tolist()
        df.iloc[index_pere, 10] = 6
        # 7 Agriculteurs et travailleurs qualifiés de l’agriculture, des pêches et des forêts (7, 34, 35)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [7, 34, 35])].tolist()
        df.iloc[index_mere, 9] = 7
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [7, 34, 35])].tolist()
        df.iloc[index_pere, 10] = 7
        # 8 Ouvriers qualifiés de l’industrie, de la construction et des artisans (8, 36, 37, 38, 39)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [8, 36, 37, 38, 39])].tolist()
        df.iloc[index_mere, 9] = 8
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [8, 36, 37, 38, 39])].tolist()
        df.iloc[index_pere, 10] = 8
        # 9 Opérateurs d'installations, de machines et ouvriers d'assemblage (9, 40, 41, 42)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [9, 40, 41, 42])].tolist()
        df.iloc[index_mere, 9] = 9
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [9, 40, 41, 42])].tolist()
        df.iloc[index_pere, 10] = 9
        # 10 Travailleurs sans compétences (10, 43, 44)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [10, 43, 44])].tolist()
        df.iloc[index_mere, 9] = 10
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [10, 43, 44])].tolist()
        df.iloc[index_pere, 10] = 10
        # 11 Métiers des Forces armées (11, 14, 15, 16)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [11, 14, 15, 16])].tolist()
        df.iloc[index_mere, 9] = 11
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [11, 14, 15, 16])].tolist()
        df.iloc[index_pere, 10] = 11
        # 12 Autre situation (12)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [12])].tolist()
        df.iloc[index_mere, 9] = 12
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [12])].tolist()
        df.iloc[index_pere, 10] = 12
        # 13 (vide) (13)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [13])].tolist()
        df.iloc[index_mere, 9] = 13
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [13])].tolist()
        df.iloc[index_pere, 10] = 13
        # 14 Directeurs des services administratifs et commerciaux (17, 18)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [17, 18])].tolist()
        df.iloc[index_mere, 9] = 14
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [17, 18])].tolist()
        df.iloc[index_pere, 10] = 14
        # 15 Professionnels de la santé (20)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [20])].tolist()
        df.iloc[index_mere, 9] = 15
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [20])].tolist()
        df.iloc[index_pere, 10] = 15
        # 16 Enseignants (21)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [21])].tolist()
        df.iloc[index_mere, 9] = 16
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [21])].tolist()
        df.iloc[index_pere, 10] = 16
        # 17 Spécialistes en finance, comptabilité, organisation administrative et relations publiques et commerciales (22)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [22])].tolist()
        df.iloc[index_mere, 9] = 17
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [22])].tolist()
        df.iloc[index_pere, 10] = 17
        # 18 Personnel de soutien administratif (27, 28, 29)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [27, 28, 29])].tolist()
        df.iloc[index_mere, 9] = 18
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [27, 28, 29])].tolist()
        df.iloc[index_pere, 10] = 18
        # 19 Assistants à la préparation des repas (45)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [45])].tolist()
        df.iloc[index_mere, 9] = 19
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [45])].tolist()
        df.iloc[index_pere, 10] = 19
        # 20 Vendeurs de rue (sauf nourriture) et prestataires de services de rue (46)
        index_mere = self.data.index[np.isin(self.data["Occupation mère"], [46])].tolist()
        df.iloc[index_mere, 9] = 20
        index_pere = self.data.index[np.isin(self.data["Occupation père"], [46])].tolist()
        df.iloc[index_pere, 10] = 20

        # On va changer les types pour les données catégoriques afin que les modèles puisse comprendre
        for feature in self.cat_feat:
            df[feature] = df[feature].astype('category')

        df.drop(["Nationalité"], axis=1, inplace=True)

        return df, list(df.select_dtypes(['category']).columns)
    
    
    
    def get_data_X_y(self, data='original', OneHot=False, Scaler=None):
        if data == 'original':
            df, cat_feat = self.get_original_dataframe()
        elif data == 'simplify':
            df, cat_feat = self.get_simplify_dataframe()
        
        categorical_feat = list(df.select_dtypes(['category']).columns)
        
        if OneHot:
            df = pd.get_dummies(df, columns=categorical_feat)

        X = df.drop(["Cible"], axis=1)
        y = df["Cible"]  
        
        if Scaler == 'MinMax':
            scaler = MinMaxScaler().fit(X)
            X = scaler.transform(X)
        elif Scaler == 'Standard':
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)

        return X, y, cat_feat
        
