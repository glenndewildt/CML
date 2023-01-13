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
    ph = priors[0]
    pnoth = sum(priors[1:])
    #dgivenh = posteriors[0]
    priopros = sum(posteriors[1:])



    dgivenh = bayesFunctionMultipleHypotheses(priors, posteriors)
    print(posteriors[0]/ (1 - posteriors[0]))


    result = (ph/ pnoth) * (dgivenh/(1 - dgivenh))
    print(posteriors[1]/posteriors[2] + posteriors[1]/posteriors[0])


    return result




print(bayesFactor([0.9,0.05,0.05],[0.2,0.6,0.2]))
