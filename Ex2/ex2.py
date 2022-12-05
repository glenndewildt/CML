import pandas as pd
import math
import numpy
import random
import matplotlib.pyplot as plt


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

    index +=1

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
plt.show()







