Écrire une bibliothèque python parcoursup permettant de transformer un fichier comme data/parcoursup/26/mp2i_26.csv en deux dataframe eleve et notes, en respectant les règles suivantes :
- Définir un devcontainer utilisant uv, marimo, pandas...
- Utiliser une variable annee (ici annee=2026)

Dans eleve, conserver seulement les colonnes suivantes :
- Candidat - Code -> code, Candidat - Nom -> nom, Candidat - Prénom -> prenom
- Sexe -> booléen fille
- Date Naissance -> annees_avance
- Profil Candidat - Libellé -> booléen terminale_france
- Candidat boursier - Libellé -> booléen boursier
- Revenu brut global -> revenu
- Distance domicile-établissement(Km) -> distance
- Demande Internat - Libellé -> booléen internat
- Niveau Classe - Libellé


Utiliser un MultiIndex pour notes avec les colonnes suivantes pour 2025/2026 (annee) et 2024/2025 (annee-1) :
- Langue vivante A scolarité - Libellé 2025/2026 -> lva
- Option facultative 1 Scolarité - Libellé 2025/2026 et Option facultative 2 Scolarité - Libellé 2025/2026 -> booléen math_expertes
- Type de contrat établissement d'origine - Libellé 2025/2026 -> booléen public
- UAI Etablissement origine 2025/2026 -> uai
- Département Etablissement origine - Code 2025/2026 -> departement
- Pays Etablissement origine - Libellé 2025/2026 -> pays
- Scolarisation sur l'année - Libellé 2025/2026 -> booléen scolarisation

Pour les notes, calculer les moyennes annuelles pour 2024/2025 et 2025/2026. Par exemple df["moyenne"]["philo"][25] doit contenir la moyenne des trois premières colonnes Effectif Classe - Philosophie - Trimestre 1 (et 2, 3). Les deux colonnes suivantes de Philosophie doivent contenir la moyenne 2026.
Faire de même pour df["rang"], df["effectif"], df["moyenne_classe"] et les matières :
- Physique-Chimie Spécialité -> pc
- Numérique et Sciences Informatiques -> nsi
- Mathématiques Expertes -> math_expertes
- Langue vivante A -> lva
- Français -> fr. Pour le français, ajouter colonne Note de l'épreuve - Français écrit -> ecrit et Note de l'épreuve - Français oral -> oral
