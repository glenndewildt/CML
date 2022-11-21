import drivermodel

tp = 100
tc = 70
tm = 70

# response times in ms
def start():
    return 0
def perceptualstep(x):
    result = 0 

    # if slow
    if(x =="slow"):
        result +=200
    elif(x =="fast"):
        result +=50
    else:
        result +=100


    return result
def cognitivestep(x):
    result = 0
    # if slow
    if(x =="slow"):
        result +=170
    elif(x =="fast"):
        result +=25
    else:
        result +=70
    return result
def motorstep(x):
    result = 0
    # if slow
    if(x =="slow"):
        result +=100
    elif(x =="fast"):
        result +=30
    else:
        result +=70
    return tm



# calculate  figure 1
def example1():
    # step 1:At	time t=perceptualstep0, the stimulus a appears .Nothing has happened in the	brain yet
    t = start()

    # step 2:peceptial process takes place (arrows around the pecetial process)
    # a is stored in the visual working memory (a')  and an impression of that os om the working memory a''
    t += perceptualstep("middle")

    # step 3: cognitive process takes place prepares motor action and yess was diceded
    t += cognitivestep("middle")
    # step 4: motor function is executed (presses)
    t += motorstep("")
    return t


print(str(example1())+ "ms")
