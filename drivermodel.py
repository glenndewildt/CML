### 
### This code is developed by Christian P. Janssen of Utrecht University
### It is intended for students from the Master's course Cognitive Modeling
### Large parts are based on the following research papers:
### Janssen, C. P., & Brumby, D. P. (2010). Strategic adaptation to performance objectives in a dual‐task setting. Cognitive science, 34(8), 1548-1560. https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01124.x
### Janssen, C. P., Brumby, D. P., & Garnett, R. (2012). Natural break points: The influence of priorities and cognitive and motor cues on dual-task interleaving. Journal of Cognitive Engineering and Decision Making, 6(1), 5-29. https://journals.sagepub.com/doi/abs/10.1177/1555343411432339
###
### If you want to use this code for anything outside of its intended purposes (training of AI students at Utrecht University), please contact the author:
### c.p.janssen@uu.nl



### 
### import packages
###
import math 
import numpy
import random
from matplotlib import pyplot as plt


###
###
### Global parameters. These can be called within functions to change (Python: make sure to call GLOBAL)
###
###

trailTime = 0
driftTimes = 0
###
### Car / driving related parameters
###
steeringUpdateTime = 250    #in ms ## How long does one steering update take? (250 ms consistent with Salvucci 2005 Cognitive Science)
timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
startingPositionInLane = 0.27 			#assume that car starts already slightly away from lane centre (in meters) (cf. Janssen & Brumby, 2010)


#parameters for deviations in car drift due the simulator environment: See Janssen & Brumby (2010) page 1555
gaussDeviateMean = 0
gaussDeviateSD = 0.13 ##in meter/sec


#When the car is actively controlled, calculate a value using equation (1) in Janssen & Brumby (2010). However, some noise is added on top of this equation to account for variation in human behavior. See Janssen & Brumby (2010) page 1555. Also see function "updateSteering" on how this function is used
gaussDriveNoiseMean = 0
gaussDriveNoiseSD = 0.1	#in meter/sec


### The car is controlled using a steering wheel that has a maximum angle. Therefore, there is also a maximum to the lateral velocity coming from a steering update
maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
minLateralVelocity = -1* maxLateralVelocity

startvelocity = 0 	#a global parameter used to store the lateral velocity of the car


###
### Switch related parameters
###
retrievalTimeWord = 200   #ms. ## How long does it take to think of the next word when interleaving after a word (time not spent driving, but drifting)
retrievalTimeSentence = 300 #ms. ## how long does it take to retrieve a sentence from memory (time not spent driving, but drifting)



###
### parameters for typing task
###
timePerWord = 0  ### ms ## How much time does one word take
wordsPerMinuteMean = 39.33   # parameters that control typing speed: when typing two fingers, on average you type this many words per minute. From Jiang et al. (2020; CHI)
wordsPerMinuteSD = 10.3 ## this si standard deviation (Jiang et al, 2020)


## Function to reset all parameters. Call this function at the start of each simulated trial. Make sure to reset GLOBAL parameters.
def resetParameters():
    global trailTime 
    global retrievalTimeWord
    global retrievalTimeSentence 
    global steeringUpdateTime 
    global startingPositionInLane 
    global gaussDeviateMean
    global gaussDeviateSD 
    global gaussDriveNoiseMean 
    global gaussDriveNoiseSD 
    global timeStepPerDriftUpdate 
    global maxLateralVelocity 
    global minLateralVelocity 
    global startvelocity
    global wordsPerMinuteMean
    global wordsPerMinuteSD

    trailTime = 0

    timePerWord = 0  ### ms

    retrievalTimeWord = 200   #ms
    retrievalTimeSentence = 300 #ms
	
    steeringUpdateTime = 250    #in ms
    startingPositionInLane = 0.27 			#assume that car starts already away from lane centre (in meters)
	

    gaussDeviateMean = 0
    gaussDeviateSD = 0.13 ##in meter/sec
    gaussDriveNoiseMean = 0
    gaussDriveNoiseSD = 0.1	#in meter/sec
    timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
    maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
    minLateralVelocity = -1* maxLateralVelocity
    startvelocity = 0 	#a global parameter used to store the lateral velocity of the car
    wordsPerMinuteMean = 39.33
    wordsPerMinuteSD = 10.3

	



##calculates if the car is not accelerating more than it should (maxLateralVelocity) or less than it should (minLateralVelocity)  (done for a vector of numbers)
def velocityCheckForVectors(velocityVectors):
    global maxLateralVelocity
    global minLateralVelocity

    velocityVectorsLoc = velocityVectors

    if (type(velocityVectorsLoc) is list):
            ### this can be done faster with for example numpy functions
        velocityVectorsLoc = velocityVectors
        for i in range(len(velocityVectorsLoc)):
            if(velocityVectorsLoc[i]>1.7):
                velocityVectorsLoc[i] = 1.7
            elif (velocityVectorsLoc[i] < -1.7):
                velocityVectorsLoc[i] = -1.7
    else:
        if(velocityVectorsLoc > 1.7):
            velocityVectorsLoc = 1.7
        elif (velocityVectorsLoc < -1.7):
            velocityVectorsLoc = -1.7

    return velocityVectorsLoc
	




## Function to determine lateral velocity (controlled with steering wheel) based on where car is currently positioned. See Janssen & Brumby (2010) for more detailed explanation. Lateral velocity update depends on current position in lane. Intuition behind function: the further away you are, the stronger the correction will be that a human makes
def vehicleUpdateActiveSteering(LD):

	latVel = 0.2617 * LD*LD + 0.0233 * LD - 0.022
	returnValue = velocityCheckForVectors(latVel)
	return returnValue
	



### function to update steering angle in cases where the driver is NOT steering actively (when they are distracted by typing for example)
def vehicleUpdateNotSteering():
    
    global gaussDeviateMean
    global gaussDeviateSD 

    

    vals = numpy.random.normal(loc=gaussDeviateMean, scale=gaussDeviateSD,size=1)[0]
    returnValue = velocityCheckForVectors(vals)
    return returnValue

updateList = []


# function that handles the drifts , parms: trailTime, IocDrift,  return : IocDrift
def handleDrift(trailTime, IocDrift, autoPostition, drifts):

        driftTimes = math.floor(trailTime / timeStepPerDriftUpdate)

        differenceDrift = driftTimes - drifts
        differenceDrift =  int(differenceDrift)

        if differenceDrift > 0:
            #foreach 50ms 
            for x in range(differenceDrift):
                #update the IocDrift with  
                update = vehicleUpdateNotSteering()
                update = update / 20
                updateList.append(update)

                autoPostition += update
                IocDrift.append(autoPostition)
                driftTimes += 1
        return driftTimes



### Function to run a trial. Needs to be defined by students (section 2 and 3 of assignment)

def runTrial(nrWordsPerSenteInitiatence =5,nrSentences=3,nrSteeringMovementsWhenSteering=2, interleaving="word"): 
    resetParameters()
    IocDrift = []
    drifts = 0
    global trailTime
    global startvelocity
    autoPosition = startingPositionInLane



    # set times per word
    WPM = numpy.random.normal(loc=39.33, scale=10.3)

    timePerWord = 60 / WPM * 1000
    #check if stratagy is word
    if interleaving == "word":
        #loop through all the sentences
        o = 0
        for  s in  range(nrSentences):
            # add time for retieving a sentce
            trailTime += retrievalTimeSentence
            drifts = handleDrift(trailTime, IocDrift, autoPosition, drifts)
            o += 0


            #loop trough all the words
            i = 0 
            for w in range(nrWordsPerSenteInitiatence):
                #add time for retrieving word
                trailTime += retrievalTimeWord
                drifts = handleDrift(trailTime, IocDrift,autoPosition, drifts)
                #add time for typing a word
                trailTime += timePerWord
                drifts = handleDrift(trailTime, IocDrift,autoPosition, drifts)
                # if not add the end update stering
                if i != nrWordsPerSenteInitiatence - 1 & o != nrSentences - 1:
                    streeings = int(nrSteeringMovementsWhenSteering)
                    #for each steering dor loop
                    for s in range(streeings):
                        # update steering time  time 
                        trailTime += steeringUpdateTime
                        # have to do in 5 steps becauNonese of the string time
                        update = vehicleUpdateActiveSteering(autoPosition)
                        update / 20
                        if autoPosition >= 0:
                            for x in range(5):
                                autoPosition -= update
                                updateList.append(update)
                                IocDrift.append(autoPosition)
                        else:
                            for x in range(5):
                                autoPosition += update
                                updateList.append(update)
                                IocDrift.append(autoPosition)
                i+=1
    #check if stratagy is sentence
    if interleaving == "sentence":
        #loop through all the sentences
        o = 0
        for  s in  range(nrSentences):
            # add time for retieving a sentce
            trailTime += retrievalTimeSentence
            drifts = handleDrift(trailTime, IocDrift, autoPosition, drifts)

            o += 0


            #loop trough all the words
            i = 0 
            for w in range(nrWordsPerSenteInitiatence):
                #add time for typing a word
                trailTime += timePerWord
                drifts = handleDrift(trailTime, IocDrift,autoPosition, drifts)

                # if not add the end update stering

                if i == nrWordsPerSenteInitiatence - 1 & o != nrSentences - 1:
                    streeings = int(nrSteeringMovementsWhenSteering)
                    #for each steering dor loop
                    for s in range(streeings):
                        # update steering time  time 
                        trailTime += steeringUpdateTime

                        # have to do in 5 steps becauNonese of the string time
                        update = vehicleUpdateActiveSteering(autoPosition)
                        update / 20
                        if autoPosition >= 0:
                            for x in range(5):
                                autoPosition -= update
                                updateList.append(update)
                                IocDrift.append(autoPosition)
                        else:
                            for x in range(5):
                                autoPosition += update
                                updateList.append(update)
                                IocDrift.append(autoPosition)
                    drifts = handleDrift(trailTime, IocDrift,autoPosition, drifts)
                i+=1


    #check if stratagy is drivingOnly
    if interleaving == "drivingOnly":
        #loop through all the sentences
        o = 0
        for  s in  range(nrSentences):
            # add time for retieving a sentce
            trailTime += retrievalTimeSentence
            o += 0


            #loop trough all the words
            i = 0 
            for w in range(nrWordsPerSenteInitiatence):
                #add time for typing a word
                trailTime += timePerWord
                # if not add the end update stering
                if i != nrWordsPerSenteInitiatence - 1 & o != nrSentences - 1:
                    streeings = int(nrSteeringMovementsWhenSteering)
                    #for each steering dor loop
                    for s in range(streeings):
                        # update steering time  time 
                        trailTime += steeringUpdateTime
                i+=1

        # have to do in 5 steps becauNonese of the string time
        for test in range(math.floor(trailTime/250)):
            update = vehicleUpdateActiveSteering(autoPosition)
            update / 20
            if autoPosition >= 0:
                for x in range(5):
                    autoPosition -= update
                    updateList.append(update)
                    IocDrift.append(autoPosition)
            else:
                for x in range(5):
                    autoPosition += update
                    updateList.append(update)
                    IocDrift.append(autoPosition)
    #check if stratagy is none
    if interleaving == "none":
        #loop through all the sentences
        o = 0
        for  s in  range(nrSentences):
            # add time for retieving a sentce
            trailTime += retrievalTimeSentence
            o += 0


            #loop trough all the words
            i = 0 
            for w in range(nrWordsPerSenteInitiatence):
                #add time for typing a word
                trailTime += timePerWord
                # if not add the end update stering
                if i != nrWordsPerSenteInitiatence - 1 & o != nrSentences - 1:
                    streeings = int(nrSteeringMovementsWhenSteering)
                    #for each steering dor loop
                    for s in range(streeings):
                        # update steering time  time 
                        trailTime += steeringUpdateTime
                i+=1
                
        # handleDrift
        handleDrift(trailTime, IocDrift, autoPosition, drifts)

    return (IocDrift , trailTime, max(IocDrift), numpy.average(IocDrift))



                
                        


### function to run multiple simulations. Needs to be defined by students (section 3 of assignment)
def runSimulations(nrSims = 100, w = 3,sen =4 , st= 2):
    sum = ["word", "sentence", "drivingOnly", "none"]
    result = []
    con = []
    maxdiv = []
    meandiv = []
    total = []
    counter = 0
    for s in sum:
      for x in range(nrSims):
        num = random.randint(5, 20)

        (d , t, maxd, meand) = runTrial(num, 10, 4, s)
        maxdiv.append(maxd)
        meandiv.append(meand)
        total.append(t)
        con.append(s)
        # print avg



        #result.append([total, maxdiv,meandiv])

    for nr in range(nrSims*4):
      x = total[nr]
      y = maxdiv[nr]
      if con[nr] == "word":
        a, = plt.plot(x, y, marker="o", markeredgecolor="grey", markerfacecolor="grey")

      if con[nr] == "sentence":
        b, = plt.plot(x, y, marker="^", markeredgecolor="grey", markerfacecolor="grey")
      if con[nr] == "drivingOnly":
        c, = plt.plot(x, y, marker="s", markeredgecolor="grey", markerfacecolor="grey")
      if con[nr] == "none":
        d, = plt.plot(x, y, marker="P", markeredgecolor="grey", markerfacecolor="grey")
    xi = math.fsum(total[0:100]) / 100
    xo = math.fsum(maxdiv[0:100]) / 100
    markersize = 10

    e, = plt.plot(xi, xo, marker="o", markeredgecolor="red",
                      markerfacecolor="red", markersize = markersize)
    plt.errorbar(xi, xo, xerr=numpy.std(total[0:100]), yerr=numpy.std(maxdiv[0:100]))

    f, = plt.plot(math.fsum(total[101:200]) / 100, math.fsum(maxdiv[101:200]) / 100, marker="^", markeredgecolor="blue",
                      markerfacecolor="blue",markersize = markersize)
    plt.errorbar(math.fsum(total[101:200]) / 100, math.fsum(maxdiv[101:200]) / 100,xerr=numpy.std(total[101:200]), yerr=numpy.std(maxdiv[101:200]))

    g, = plt.plot(math.fsum(total[201:300]) / 100, math.fsum(maxdiv[201:300]) / 100, marker="s", markeredgecolor="green",
                      markerfacecolor="green",markersize = markersize)
    plt.errorbar(math.fsum(total[201:300]) / 100, math.fsum(maxdiv[201:300]) / 100, xerr=numpy.std(total[201:300]), yerr=numpy.std(maxdiv[201:300]))

    h, = plt.plot(math.fsum(total[301:400]) / 100, math.fsum(maxdiv[301:400]) / 100, marker="P", markeredgecolor="yellow",
                      markerfacecolor="yellow",markersize = markersize)
    plt.errorbar(math.fsum(total[301:400]) / 100, math.fsum(maxdiv[301:400]) / 100, xerr=numpy.std(total[301:400]), yerr=numpy.std(maxdiv[301:400]), markeredgecolor = "yellow")

    print("std max div: "+ str(numpy.std(maxdiv[0:100])))
    print("std total time: "+str(numpy.std(total[0:100])))
    print(xi)
    print(xo)


    print("std max div: "+ str(numpy.std(maxdiv[101:200])))
    print("std total time: "+str(numpy.std(total[101:200])))
    print(math.fsum(total[101:200]) / 100)
    print(math.fsum(maxdiv[101:200]) / 100)


    plt.legend([e, f, g, h], ["Average word", "Average sentence", "Average drivingOnly", "Average none"])
    plt.xlabel("Time (in ms)")

    plt.ylabel("Lane position (in m)")
    plt.show()

    counter +=1


    #return result

def getMaxUpdate():
    return max(updateList)

def getMeanUpdate():
    up = []
    for u in updateList:
        up.append(abs(u))
    return numpy.average(u)

def getUpdate():
    up = []
    for u in updateList:
        up.append(abs(u))
    return u

# check for last word/sentence was incorrect
# removed endSentence part as we dont use it
# think autoPosition check (after vehicleUpdateActiveSteering(autoPosition)) should be outside for loop
# left axis should be in meters
# Asume steering in drivingOnly does still take 250 ms.
# Do we use updateList?

runSimulations()
