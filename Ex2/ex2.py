import pandas as pd
import math
import numpy
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import sem
import statistics
import matplotlib.ticker as ticker

import researchpy as rp



nr1 = pd.read_csv('NeuralResponses_S1.txt')
nr2 = pd.read_csv('NeuralResponses_S2.txt')
catVectors = pd.read_csv('CategoryVectors.txt')
catLabels = pd.read_csv('CategoryLabels.txt')
beh = pd.read_csv('BehaviourRDM.csv')


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


avgResPos =[]
for r in PosResult:
    avgResPos.append(numpy.average(r))


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
  templist.append(avgResPos[n] - avgPos[n])

# Build the plot
fig, ax = plt.subplots()
ax.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],templist[0:20], ecolor='black')
plt.rcParams["figure.autolayout"] = True
# Be sure to only pick integer tick locations.
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

ax.set_ylabel('Response Aplitude')
ax.set_title('Voxels')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()

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

avgBoth = []

for i in range(len(avgPos)):
    x = numpy.average(avgPos[i])
    y = numpy.average(avgZero[i])

    avgBoth.append(x-y)



cats = ["animate","inanimate"]


# Calculate the average
Pos_mean = numpy.mean(avgPos)
Zero_mean = numpy.mean(avgZero)
avgBoth.append(abs(Pos_mean - Zero_mean))
Both_mean = numpy.mean(avgBoth)

# Calculate the standard deviation
Pos_std = numpy.std(avgPos)
#x = abs(avgPos - Pos_mean)**2
#Pos_std = math.sqrt(numpy.mean(x))
Zero_std = numpy.std(avgZero)
#Zero_std = math.sqrt(abs(Zero_mean))
Both_std = numpy.std((numpy.subtract(avgZero , avgPos)))
#Both_std = math.sqrt(abs(Both_mean))

# Create lists for the plot
x_pos = numpy.arange(len(cats))
CTEs = [Pos_mean, Zero_mean]
error = [sem(avgPos), sem(avgZero)]#error = [Pos_std, Zero_std]

y = abs((Zero_mean - avgPos) - (Pos_mean - Zero_mean))**2
stdtest = math.sqrt(numpy.mean(y))

test = (Both_mean)/((Both_std)/(numpy.sqrt(44)))#2.4737
print("test")
print(stats.t.sf(1.5822132471058494, df=40)*2)

stats.t.ppf(avgPos, 43)


# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Response Amplitude')
ax.set_xticks(x_pos)
ax.set_xticklabels(cats)
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


cats = ["animate","inanimate"]

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

plt.show()

fig, ax = plt.subplots()


ax.scatter(numpy.resize(clf.coef_,20),templist[0:20])
ax.set_ylabel('Responses')
ax.set_xlabel('Weights')

plt.show()

print(numpy.corrcoef(numpy.resize(clf.coef_,20),templist[0:20]))


TrainHumanNonHuman = resultPos[0 : 10] + resultPos[24 : 34]
TestHumanNonHuman = resultPos[10 : 20] + resultPos[34 : 44]


plt.show()

zeros = [0] * 10
ones = [1] * 10

TrainLabels = zeros + ones

clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(TrainHumanNonHuman, TrainLabels)
prediction = clf.predict(TestHumanNonHuman)
print("Accuracy:",metrics.accuracy_score(TrainLabels, prediction))

########################################## 3


numpy.corrcoef(clf.coef_,templist)

AllImages = []
index = 0

for value in catVectors.iloc[:,0]:

    AllImages.append(nr1.iloc[index, :])
    index += 1

print(len(AllImages))


heatmap = numpy.arange(88*88).reshape(88,88)
image_index = 0
# for image in AllImages :
#     compare_index = 0
#
#     for compare in AllImages:
#         heatmap[image_index][compare_index] = numpy.corrcoef(image, compare)[0][1]
#
#         #print(numpy.subtract( 1,numpy.corrcoef(image, compare)))
#         compare_index += 1
#     image_index +=1
import seaborn as sns

heatmap = numpy.corrcoef(AllImages)

heatmap = numpy.subtract(1,heatmap)
#heatmap = numpy.multiply(-1,heatmap)

print(heatmap)

print(len(heatmap))
ax = sns.heatmap(heatmap)


plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.show()

heatmap2 = heatmap[~numpy.eye(heatmap.shape[0],dtype=bool)].reshape(heatmap.shape[0],-1)

GroupA = heatmap2[0:44, 0:43]
GroupA1 = heatmap2[44:88, 44:87]

GroupAFinal = numpy.divide(numpy.add(GroupA1, GroupA), 2)
print("numpy.shape(GroupAFinal)")

numpy.shape(GroupAFinal)


GroupB = heatmap2[0:44, 44:88]


avgGroupA = []
avgGroupB = []

for i in range(len(avgPos)):
    avgGroupA.append(numpy.average(GroupAFinal[i]))
    avgGroupB.append(numpy.average(GroupB[i]))






print("Tetset =  "+str(stats.ttest_ind(a= avgGroupA, b=avgGroupB)))


GroupA = numpy.average(GroupAFinal)

GroupB = numpy.average(heatmap2[0:44, 44:88])
print("numpy.shape(GroupB)")

err1 = sem(avgGroupA)
err2 = sem(avgGroupB)
print(err1)
print(err2)

# Build the plot
fig, ax = plt.subplots()
ax.bar([0,1],[GroupA, GroupB], yerr=[err1, err2], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(["same","different"])
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()

print(beh.iloc[::])
print(numpy.shape(beh.iloc[::]))

plt.imshow(beh.iloc[::], cmap='hot', interpolation='nearest')
plt.show()

beh2 = numpy.array(beh.iloc[~numpy.eye(beh.iloc[::].shape[0],dtype=bool)]).reshape(beh.iloc[::].shape[0],-1)
print(numpy.average(heatmap))
print(numpy.average(beh.iloc[::]))
print(numpy.average(beh2[0:44, 44:87]))
print(numpy.average(heatmap2[0:44, 44:87]))
print(numpy.average(numpy.divide(beh2[0:44, 0:43]+ beh2[44:88, 44:87], 2)))
print(numpy.average(GroupAFinal))





print(numpy.average(numpy.corrcoef(beh.iloc[::],heatmap)))
print(numpy.average(numpy.corrcoef(beh2[0:44, 44:87],heatmap2[0:44, 44:87])))
print(numpy.average(numpy.corrcoef(numpy.divide(beh2[0:44, 0:43]+ beh2[44:88, 44:87], 2),GroupAFinal)))
plt.imshow(numpy.corrcoef(beh.iloc[::],heatmap), cmap='hot', interpolation='nearest')
plt.show()







