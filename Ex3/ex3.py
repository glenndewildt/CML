import numpy as np
import math


def bayesFunction(h, dh, dnoth):
    return (h * dh) / ((dh * h) + (dnoth* (1-h)))


def bayesFunctionMultipleHypotheses(priors , likelihood):
    """
    priors = A	vector	of	prior	probabilities	of	all	possible	hypotheses.
    likelihood = A	vector	of	all	likelihood	functions	of	the	data	given	these	hypotheses.
    """
    result = 0
    for i in range(len(priors)):
        result += priors[i] * likelihood[i]

    return (priors[0] * likelihood[0]) / result

def bayesFactor(posteriors, priors):
    # ph = priors[0]
    # pnoth = sum(priors[1:])
    # #dgivenh = posteriors[0]
    # priopros = sum(posteriors[1:])



    # dgivenh = bayesFunctionMultipleHypotheses(priors, posteriors)
    # print(posteriors[0]/ (1 - posteriors[0]))


    # result = (ph/ pnoth) * (dgivenh/(1 - dgivenh))
    # print(posteriors[1]/posteriors[2] + posteriors[1]/posteriors[0])
    ph1d = posteriors[0]
    ph1 = priors[0]
    result2 = []
    
    for i in range(len(priors)):
        if i == 0:
            ph2 = 1 - priors[i]
            ph2d = 1 - posteriors[i]
        else:
            ph2 = priors[i]
            ph2d = posteriors[i]
        result = (ph1d*ph2)/(ph2d*ph1)
        result2.append(result)

    return result2




#print(bayesFactor([0.9,0.05,0.05],[0.2,0.6,0.2]))
posteriors = bayesFunctionMultipleHypotheses([0.5,0.5],[0.531,0.52])
#print(posteriors)
priors = [0.5,0.5]
#print(bayesFactor([posteriors, 1-posteriors], priors))
priors = [0.001,0.999]
posteriors = bayesFunctionMultipleHypotheses(priors,[0.531,0.52])
#print(posteriors)
#print(bayesFactor([posteriors, 1-posteriors], priors))

prior = bayesFunctionMultipleHypotheses([0.5,0.5],[0.531,0.52])
priors = [prior , 1- prior]
likelihood = [0.471, 0.520]
posteriors = bayesFunctionMultipleHypotheses(priors,likelihood)

print(posteriors)

priors = [posteriors , 1- posteriors]
likelihood = [0.491, 0.65]
posteriors = bayesFunctionMultipleHypotheses(priors,likelihood)

print(posteriors)

priors = [posteriors , 1- posteriors]
likelihood = [0.505, 0.70]
posteriors = bayesFunctionMultipleHypotheses(priors,likelihood)
b1 = bayesFunction(0.411, 0.505,0.70)
b2 = bayesFunction(0.589, 0.505,0.70)

bf = bayesFactor([b1,b2], [0.505,0.70])

print(bf)
print(posteriors)


