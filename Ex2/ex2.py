import pandas as pd
import math
import numpy
import random
import matplotlib.pyplot as plt
from scipy import stats



nr1 = pd.read_csv('NeuralResponses_S1.txt')
nr2 = pd.read_csv('NeuralResponses_S2.txt')
catVectors = pd.read_csv('CategoryVectors.txt')
catLabels = pd.read_csv('CategoryLabels.txt')

resultPos = []
resultZero = []

index = 0

for value in catVectors.iloc[:,0]:

    if value == 0:
        resultZero.append(nr1.iloc[index, :])

    if value == 1:
        resultPos.append(nr1.iloc[index, :])

    index += 1

##########################################
### 1 B
##########################################
index = 0
PosResult = []
for x in range(100):
    PosResult.append([])
for cat in resultPos:
    index = 0
    for c in cat:
        PosResult[index].append(c)
        index += 1



avgRes =[]
for r in PosResult:
    avgRes.append(numpy.average(r))


ZeroResult = []
for x in range(100):
    ZeroResult.append([])
for cat in resultZero:
    index = 0

    for c in cat:
        ZeroResult[index].append(c)
        index += 1

avgPos =[]
for r in ZeroResult:
    avgPos.append(numpy.average(r))


templist = []

for n in range(100):
  templist.append(avgRes[n] - avgPos[n])

# Build the plot
fig, ax = plt.subplots()
ax.bar(numpy.arange(100),templist, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Response Aplitude')
ax.set_title('Voxels')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.show()

##########################################
### 1 A
##########################################
avgPos = []
for cat in resultPos:
    x = numpy.average(cat)
    avgPos.append(x)

avgZero = []

for cat in resultZero:
    x = numpy.average(cat)
    avgZero.append(x)

cats = ["an","inAn"]

# Calculate the average
Pos_mean = numpy.mean(avgPos)
Zero_mean = numpy.mean(avgZero)

# Calculate the standard deviation
Pos_std = numpy.std(avgPos)
Zero_std = numpy.std(avgZero)


# Create lists for the plot
x_pos = numpy.arange(len(cats))
CTEs = [Pos_mean, Zero_mean]
error = [Pos_std, Zero_std]
test = (abs(Pos_mean - Zero_mean))/((abs(Pos_std - Zero_std))/numpy.sqrt(44))

stats.t.ppf(avgPos, 43)


# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Label y')
ax.set_xticks(x_pos)
ax.set_xticklabels(cats)
ax.set_title('label x')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.show()

##################################
#### 2A
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

resultPos = []
resultZero = []

index = 0

for value in catVectors.iloc[:,0]:

    if value == 0:
        resultZero.append(nr2.iloc[index, :])

    if value == 1:
        resultPos.append(nr2.iloc[index, :])

    index += 1


cats = ["an","inAn"]

trainingSetAn= resultPos[0 : 22]
TestSetAn= resultPos[22 : 44]
trainingSetInAn= resultZero[0 : 22]
TestSetInAn= resultZero[22 : 44]

vectorizer = CountVectorizer()
zeros = [0] * 22
ones = [1] * 22

TrainLabels = zeros + ones

TrainAB = trainingSetAn + trainingSetInAn
TestAB = TestSetAn + TestSetInAn








clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(TrainAB, TrainLabels)
prediction = clf.predict(TestAB)
print("Accuracy:",metrics.accuracy_score(TrainLabels, prediction))






plt.scatter(templist, clf.coef_)
plt.show()





