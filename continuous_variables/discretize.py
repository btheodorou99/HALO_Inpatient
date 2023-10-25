import pickle

trainData = pickle.load(open('data/trainDataset.pkl', 'rb'))
valData = pickle.load(open('data/valDataset.pkl', 'rb'))

idToLab = pickle.load(open('data/idx_to_lab.pkl', 'rb'))
labToNumber = {l: i for (i,l) in enumerate(pickle.load(open('data/id_to_channel.pkl', 'rb')))}
isCategorical = pickle.load(open('data/is_categorical_channel.pkl', 'rb'))
beginPos = pickle.load(open('data/begin_pos.pkl', 'rb'))
possibleValues = pickle.load(open('data/possible_values.pkl', 'rb'))
variableRanges = pickle.load(open('data/variable_ranges.pkl', 'rb'))

discretization = {
    'Diastolic blood pressure': [0, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 130, 375],
    'Fraction inspired oxygen': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.001, 1.1],
    'Glucose': [0, 40, 60, 80, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 225, 275, 325, 400, 600, 800, 1000, 2200],
    'Heart Rate': [0, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 390],
    'Height': [0, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 230],
    'Mean blood pressure': [0, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 375],
    'Oxygen saturation': [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100, 100.001, 150],
    'pH': [6.3, 6.7, 7.1, 7.35, 7.45, 7.6, 8.0, 8.3, 10],
    'Respiratory rate': [0, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 330],
    'Systolic blood pressure': [0, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 230, 375],
    'Temperature': [14.2, 30, 32, 33, 33.5, 34, 34.5, 35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 40, 47],
    'Weight': [0, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 160, 170, 190, 210, 250],
    'Age': [18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
    'Days': [0, 11, 16, 21, 25, 30.1, 35.1, 43, 48, 54, 60, 66, 72, 81, 90, 100.1],
    'Hours': [0, 0.5, 1.5, 2.5, 3.5, 6.5, 10.5, 16.5, 26.5, 48.0, 48.1, 60.1, 80.1, 110.1, 150.1, 200.1]
}

formatMap = {
    'Diastolic blood pressure': ('.0f', int),
    'Fraction inspired oxygen': ('.2f', float),
    'Glucose': ('.0f', int),
    'Heart Rate': ('.0f', int),
    'Height': ('.0f', int),
    'Mean blood pressure': ('.0f', int),
    'Oxygen saturation': ('.0f', int),
    'pH': ('.2f', float),
    'Respiratory rate': ('.0f', int),
    'Systolic blood pressure': ('.0f', int),
    'Temperature': ('.1f', float),
    'Weight': ('.1f', float),
    'Age': ('.2f', float),
    'Days': ('.2f', float),
    'Hours': ('.1f', float)
}

def get_index(mapping, key, value):
    possible_values = mapping[key]
    for i in range(len(possible_values) - 1):
        if value < possible_values[i + 1]:
            return i
    
    print(f"{value} for {key} not in {possible_values}")
    return len(possible_values) - 2

# Convert to New Data Format
for p in (trainData + valData):
    new_visits = []
    firstVisit = True
    for v in p['visits']:
        if v[1] == []:
            new_cont = get_index(discretization, 'Age' if firstVisit else 'Days', v[3][-1])
            firstVisit = False
            new_visits.append((v[0], [], [], [new_cont]))
        else:
            new_labs = []
            new_values = []
            for l, val in zip(v[1], v[2]):
                if isCategorical[idToLab[l]]:
                    if val == 1:
                        new_labs.append(labToNumber[idToLab[l]])
                        new_values.append(beginPos[labToNumber[idToLab[l]]] - l)
                else:
                    if val < variableRanges[idToLab[l]][0] or val >= variableRanges[idToLab[l]][1]:
                        continue
                    
                    new_labs.append(labToNumber[idToLab[l]])
                    new_values.append(get_index(discretization, idToLab[l], val))

            if not new_labs:
                continue
            new_cont = get_index(discretization, 'Hours', v[3][-1])
            new_visits.append((v[0], new_labs, new_values, [new_cont]))
    
    p['visits'] = new_visits

pickle.dump(trainData, open('discretized_data/trainDataset.pkl', 'wb'))
pickle.dump(valData, open('discretized_data/valDataset.pkl', 'wb'))

newIdToLab = {i:l for (l,i) in labToNumber.items()}
newBeginPos = []
seenContinuous = False
for i in range(len(newIdToLab)):
    if not seenContinuous:
        newBeginPos.append(beginPos[i])
        if not isCategorical[newIdToLab[i]]:
            seenContinuous = True
            currPos = newBeginPos[i] + len(discretization[newIdToLab[i]]) - 1   
    else:
        newBeginPos.append(currPos)
        currPos += len(discretization[newIdToLab[i]]) - 1   

newIdxToId = {}
for i in range(len(newBeginPos) - 1):
    for j in range(newBeginPos[i], newBeginPos[i+1]):
        newIdxToId[j] = i
for j in range(newBeginPos[-1], newBeginPos[-1] + len(discretization[newIdToLab[len(newBeginPos) - 1]]) - 1):
    newIdxToId[j] = len(newBeginPos) - 1

pickle.dump(newIdxToId, open('discretized_data/idxToId.pkl', 'wb'))
pickle.dump(formatMap, open('discretized_data/formatMap.pkl', 'wb'))
pickle.dump(newIdToLab, open('discretized_data/idToLab.pkl', 'wb'))
pickle.dump(newBeginPos, open('discretized_data/beginPos.pkl', 'wb')) 
pickle.dump(isCategorical, open('discretized_data/isCategorical.pkl', 'wb')) 
pickle.dump(possibleValues, open('discretized_data/possibleValues.pkl', 'wb'))
pickle.dump(discretization, open('discretized_data/discretization.pkl', 'wb'))

print(f"NUM LABS: {newBeginPos[-1] + len(discretization[newIdToLab[16]]) - 1}")
print(f"NUM CONTINUOUS: {len(discretization['Age']) - 1}")
