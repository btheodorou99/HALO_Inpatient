import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml
import pickle
from sklearn.model_selection import train_test_split

MAX_TIME_STEPS = 150

mimic_dir = "./data/"
timeseries_dir = "../../Code/mimic3-benchmarks/data/root/all/"
#valid_subjects = os.listdir(timeseries_dir)
patientsFile = mimic_dir + 'PATIENTS.csv'
admissionFile = mimic_dir + "ADMISSIONS.csv"
diagnosisFile = mimic_dir + "DIAGNOSES_ICD.csv"
procedureFile = mimic_dir + "PROCEDURES_ICD.csv"
medicationFile = mimic_dir + "PRESCRIPTIONS.csv"

channel_to_id = pickle.load(open(mimic_dir + 'channel_to_id.pkl', 'rb'))
is_categorical_channel = pickle.load(open(mimic_dir + 'is_categorical_channel.pkl', 'rb'))
possible_values = pickle.load(open(mimic_dir + 'possible_values.pkl', 'rb'))
begin_pos = pickle.load(open(mimic_dir + 'begin_pos.pkl', 'rb'))
end_pos = pickle.load(open(mimic_dir + 'end_pos.pkl', 'rb'))
    
print("Loading CSVs Into Dataframes")
patientsDf = pd.read_csv(patientsFile, dtype=str).set_index("subject_id")
patientsDf = patientsDf[['gender', 'dob']]
patientsDf['dob'] = pd.to_datetime(patientsDf['dob'])
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['admittime'] = pd.to_datetime(admissionDf['admittime'])
admissionDf = admissionDf.sort_values('admittime')
admissionDf = admissionDf.reset_index(drop=True)
diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("hadm_id")
diagnosisDf = diagnosisDf[diagnosisDf['icd9_code'].notnull()]
diagnosisDf = diagnosisDf[['icd9_code']]
procedureDf = pd.read_csv(procedureFile, dtype=str).set_index("hadm_id")
procedureDf = procedureDf[procedureDf['icd9_code'].notnull()]
procedureDf = procedureDf[['icd9_code']]
medicationDf = pd.read_csv(medicationFile, dtype=str).set_index("hadm_id")
medicationDf = medicationDf[medicationDf['ndc'].notnull()]
medicationDf = medicationDf[medicationDf['ndc'] != 0]
medicationDf = medicationDf[['ndc', 'drug']]
medicationDf['ndc'] = medicationDf['ndc'].astype(int).astype(str)
medicationDf['ndc'] = [('0' * (11 - len(c))) + c for c in medicationDf['ndc']]
medicationDf['ndc'] = [c[0:5] + '-' + c[5:9] + '-' + c[10:12] for c in medicationDf['ndc']]

print("Building Dataset")
data = {}
for row in tqdm(admissionDf.itertuples(), total=len(admissionDf)):          
    hadm_id = row.hadm_id
    subject_id = row.subject_id
    admit_time = row.admittime
    
    if subject_id not in patientsDf.index:
        continue
      
    visit_count = (0 if subject_id not in data else len(data[subject_id]['visits'])) + 1
      
    tsDf = pd.read_csv(f"{timeseries_dir}{subject_id}/episode{visit_count}_timeseries.csv") if os.path.exists(f"{timeseries_dir}{subject_id}/episode{visit_count}_timeseries.csv") else None
    
    # Extract the gender and age
    patientRow = patientsDf.loc[[subject_id]].iloc[0]
    age = (admit_time.to_pydatetime() - patientRow['dob'].to_pydatetime()).days / 365
    if age > 120:
        continue
            
    # Extracting the Diagnoses
    if hadm_id in diagnosisDf.index: 
        diagnoses = list(set(diagnosisDf.loc[[hadm_id]]["icd9_code"]))
    else:
        diagnoses = []
    
    # Extracting the Procedures
    if hadm_id in procedureDf.index: 
        procedures = list(set(procedureDf.loc[[hadm_id]]["icd9_code"]))
    else:
        procedures = []
        
    # Extracting the Medications
    if hadm_id in medicationDf.index: 
        medications = list(set(medicationDf.loc[[hadm_id]]["ndc"]))
    else:
        medications = []
        
    # Extract the lab timeseries
    labs = []
    prevTime = 0
    currTime = int(tsDf.iloc[0]['Hours']) if tsDf is not None else 0
    currMask = []
    currValues = []
    if tsDf is not None:
        for i, row in tsDf.iterrows():
            rowTime = int(row['Hours'])
            
            if rowTime != currTime:
                labs.append((currMask, currValues, [currTime - prevTime]))
                prevTime = currTime
                currTime = rowTime
                currMask = []
                currValues = []
            
            for col, value in row.iteritems():
                if value != value or col == 'Hours':
                    continue
            
                if is_categorical_channel[col]:
                    if col == 'Glascow coma scale total':
                        value = str(int(value))
                    elif col == 'Capillary refill rate':
                        value = str(value)

                    if begin_pos[channel_to_id[col]] in currMask:
                        currValues[currMask.index(begin_pos[channel_to_id[col]] + possible_values[col].index(value))] = 1
                    else:
                        for j in range(begin_pos[channel_to_id[col]], end_pos[channel_to_id[col]]):
                            currMask.append(j)
                            currValues.append(1 if j - begin_pos[channel_to_id[col]] == possible_values[col].index(value) else 0)
                else:
                    if begin_pos[channel_to_id[col]] in currMask:
                        currValues[currMask.index(begin_pos[channel_to_id[col]])] = value
                    else:
                        currMask.append(begin_pos[channel_to_id[col]])
                        currValues.append(value)
      
        labs.append((currMask, currValues, [currTime - prevTime]))

    # Building the hospital admission data point
    if subject_id not in data:
        data[subject_id] = {'visits': [(diagnoses, procedures, medications, age, labs)]}
    else:
        data[subject_id]['visits'].append((diagnoses, procedures, medications, age, labs))

# Build the label mapping
print("Adding Labels")
with open("../hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
    definitions = yaml.full_load(definitions_file)

code_to_group = {}
for group in definitions:
    if definitions[group]['use_in_benchmark'] == False:
        continue
    codes = definitions[group]['codes']
    for code in codes:
        if code not in code_to_group:
            code_to_group[code] = group
        else:
            assert code_to_group[code] == group

id_to_group = sorted([k for k in definitions.keys() if definitions[k]['use_in_benchmark'] ==  True])
group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

# Add Labels
for p in data:
    label = np.zeros(len(group_to_id))
    for v in data[p]['visits']:
        for d in v[0]:
            d = str(d)
            if d not in code_to_group:
                continue

            label[group_to_id[code_to_group[d]]] = 1
    
    data[p]['labels'] = label


# Convert diagnoses, procedures, and medications to text
print("Converting Codes to Text")
medMapping = {row['ndc']: row['drug'] for _, row in medicationDf.iterrows()}
for p in data:
    new_visits = []
    for v in data[p]['visits']:
        new_visit = []
        for c in v[0]:
            new_visit.append(c)
        for c in v[1]:
            new_visit.append(c)
        for c in v[2]:
            if c in medMapping:
                new_visit.append(medMapping[c])
            else:
                new_visit.append(c)
        
        new_visits.append((new_visit, [], [], [v[3]]))
        
        for lab_v in v[4]:
            new_visits.append(([], lab_v[0], lab_v[1], lab_v[2]))
    data[p]['visits'] = new_visits    


# Convert diagnoses, procedures, and medications to indices
print("Converting Codes to Indices")
allCodes = list(set([c for p in data for v in data[p]['visits'] for c in v[0]]))
np.random.shuffle(allCodes)
code_to_index = {c: i for i, c in enumerate(allCodes)}
counter = 0
for p in data:
    new_visits = []
    for v in data[p]['visits']:
        new_visit = []
        for c in v[0]:
            new_visit.append(code_to_index[c])
                
        new_visits.append((new_visit, v[1], v[2], v[3]))
    data[p]['visits'] = new_visits
    
index_to_code = {v: k for k, v in code_to_index.items()}
data = list(data.values())
data = [{'labels': p['labels'], 'visits': p['visits'][:MAX_TIME_STEPS - 2]} for p in data] # 2 for the start and label visits

# Train-Val-Test Split
print("Splitting Datasets")
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

# Save Everything
print("Saving Everything")
print(f"CODE VOCAB SIZE: {len(index_to_code)}")
print(f"LABEL VOCAB SIZE: {len(data[0]['labels'])}")
pickle.dump(dict((i, x) for (x, i) in list(group_to_id.items())), open("./data/idToLabel.pkl", "wb"))
pickle.dump(index_to_code, open("./data/indexToCode.pkl", "wb"))
pickle.dump(data, open("./data/allData.pkl", "wb"))
pickle.dump(train_dataset, open("./data/trainData.pkl", "wb"))
pickle.dump(val_dataset, open("./data/valData.pkl", "wb"))
pickle.dump(test_dataset, open("./data/testData.pkl", "wb"))
