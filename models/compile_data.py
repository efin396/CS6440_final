from datetime import date

import pandas as pd
from tensorflow_decision_forests.keras import (
    pd_dataframe_to_tf_dataset,
    RandomForestModel
)
from sklearn.model_selection import train_test_split

MODEL_OUTPUT_PATH = "./models/decision_forests/df_model"

# Step 1: Import patients data and conditions data
PATIENTS = '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/data/csv/combined_patients.csv'
CONDITIONS = '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/data/csv/combined_conditions.csv'

patients = pd.read_csv(PATIENTS)
conditions = pd.read_csv(CONDITIONS)

# Step 2: determine which fields you want to cherrypick from each dataset
patients_columns = [
    'Id',
    'BIRTHDATE',
    'DEATHDATE',
    'MARITAL',
    'RACE',
    'ETHNICITY',
    'GENDER',
    'INCOME'
]

# Step 3: Select columns for combination
master_df = patients[patients_columns]
opiate_dependent = conditions[conditions['CODE'].isin([55680006, 6525002])]  # This code signifies drug overdose risk
chronic_pain = conditions[conditions['CODE'].isin([82423001])]  # Chronic Pain
chronic_migraine = conditions[conditions['CODE'].isin([124171000119105])]
imapcted_molars = conditions[conditions['CODE'].isin([196416002])]

master_df['DEATHDATE'].fillna(pd.to_datetime(date.today()), inplace=True)

master_df['BIRTHDATE'] = pd.to_datetime(master_df['BIRTHDATE'])
master_df['DEATHDATE'] = pd.to_datetime(master_df['DEATHDATE'])

master_df['AGE'] = (master_df['DEATHDATE'] - master_df['BIRTHDATE']).dt.days // 365
master_df.drop(columns=['BIRTHDATE', 'DEATHDATE'], inplace=True)

opiate_ids = set(opiate_dependent['PATIENT'])
pain_ids = set(chronic_pain['PATIENT'])
migraine_ids = set(chronic_migraine['PATIENT'])
molar_ids = set(imapcted_molars['PATIENT'])

master_df['CHRONIC_PAIN'] = master_df['Id'].isin(pain_ids).astype(int)
master_df['CHRONIC_MIGRAINE'] = master_df['Id'].isin(migraine_ids).astype(int)
master_df['IMPACTED_MOLARS'] = master_df['Id'].isin(molar_ids).astype(int)
master_df['DEPENDENT'] = master_df['Id'].isin(opiate_ids).astype(int)

master_df['MARITAL'].fillna('N', inplace=True)

master_df.drop(columns=['Id'], inplace=True)

train_df, test_df = train_test_split(master_df, test_size=0.2, random_state=143)
