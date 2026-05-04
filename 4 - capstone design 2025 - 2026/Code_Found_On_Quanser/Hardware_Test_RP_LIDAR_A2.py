from Quanser.q_essential import LIDAR
import time
import struct
import numpy as py
import matplotlib.pyplot as plt

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 30
sampleTime = 1/sampleRate
simulationTime = 30.0
print(' Sample Time: ', sampleTime)

counter = 0

myLidar = LIDAR(num_measurements=7200)

h = plt.polar()
plt.show(block=False)

try:
    while elapsed_time() < simulationTime:

        plt.cla()

        start = time.time()

        myLidar.read()

        if counter % 10 == 1:
            plt.scatter(myLidar.angles, myLidar.distances, marker='.')
            plt.draw()
            print(myLidar.distances)
            plt.pause(.0001)


        end = time.time()

        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime)

        time.sleep(sleepTime)
        counter += 1
        print('Simulation Timestamp :', elapsed_time(), ' s.')

except KeyboardInterrupt:
    print("User interrupted!")
    
finally:
    plt.close()
    
    myLidar.terminate()
    
