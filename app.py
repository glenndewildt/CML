import drivermodel
import matplotlib.pyplot as plt

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

    return result



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


# Q 1D
def example2(x):
    print("hello")
    if(x == "extreme"):
        t1 = start()
        t1 += perceptualstep("fast")
        t1 += cognitivestep("fast") 
        t1 += motorstep("fast")
        print(t1)


        t2 = start()
        t2 += perceptualstep("middle")
        t2 += cognitivestep("middle")
        t2 += motorstep("middle")
        print(t2)

        t3 = start()
        t3 += perceptualstep("slow")
        t3 += cognitivestep("slow")
        t3 += motorstep("slow")
        print(t3)
        return [t1,t2,t3]
    elif(x == "all"):
        a = start()
        fast_array = []
        fast_array.append(perceptualstep("fast"))
        fast_array.append(perceptualstep("middle")) 
        fast_array.append(perceptualstep("slow")) 

        middle_array = []
        middle_array.append(cognitivestep("fast"))
        middle_array.append(cognitivestep("middle")) 
        middle_array.append(cognitivestep("slow")) 

        slow_array = []
        slow_array.append(motorstep("fast"))
        slow_array.append(motorstep("middle")) 
        slow_array.append(motorstep("slow")) 
        result = []
        for f in fast_array:
            for m in middle_array:
                for s in slow_array:
                    result.append(f + m + s)
        return result

def example3(x, y, z):
    t1 = start()
    t1 += perceptualstep(x)
    t1 += perceptualstep(x)
    t1 += cognitivestep(y)
    t1 += cognitivestep(y)
    t1 += motorstep(z)

    return t1

      


print(example2("all"))
plt.boxplot(example2("all"))
plt.show()
print(example3("fast", "fast", "slow"))
