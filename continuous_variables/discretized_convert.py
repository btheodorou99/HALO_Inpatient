import pickle
import random

idToLab = pickle.load(open('discretized_data/idToLab.pkl', 'rb'))
isCategorical = pickle.load(open('discretized_data/isCategorical.pkl', 'rb'))
discretization = pickle.load(open('discretized_data/discretization.pkl', 'rb'))
possibleValues = pickle.load(open('discretized_data/possibleValues.pkl', 'rb'))
discretization = pickle.load(open('discretized_data/discretization.pkl', 'rb'))
formatMap = pickle.load(open('discretized_data/formatMap.pkl', 'rb'))
idToLabel = pickle.load(open('discretized_data/idToLabel.pkl', 'rb'))
indexToCode = pickle.load(open('discretized_data/indexToCode.pkl', 'rb'))

dataset = pickle.load(open('results/datasets/haloDataset.pkl', 'rb'))

def formatCont(value, key):
  return formatMap[key][1](("{:" + formatMap[key][0] + "}").format(value))

for p in dataset:
  new_visits = []
  firstVisit = True
  for v in p['visits']:
    new_labs = []
    new_values = []
    for i in range(len(v[1])):
      new_labs.append(idToLab[v[1][i]])
      if isCategorical[idToLab[v[1][i]]]:
        new_values.append(possibleValues[idToLab[v[1][i]]][v[2][i]])
      else:
        new_values.append(formatCont(random.uniform(discretization[idToLab[v[1][i]]][v[2][i]], discretization[idToLab[v[1][i]]][v[2][i]+1]), idToLab[v[1][i]]))
    contType = 'Hours' if new_labs != [] else 'Age' if firstVisit else 'Days'
    if contType == 'Age':
      firstVisit = False
    new_cont = formatCont(random.uniform(discretization[contType][v[4][-1]], discretization[contType][v[4][-1]+1]), contType)
    new_visits.append((v[0], new_labs, new_values, [new_cont]))
  p['visits'] = new_visits

pickle.dump(dataset, open('results/datasets/haloDataset_converted.pkl', 'wb'))
