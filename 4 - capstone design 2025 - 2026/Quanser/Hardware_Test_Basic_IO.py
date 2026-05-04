from Quanser.product_QCar import QCar
import time
import struct
import numpy as np 

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 1000
sampleTime = 1/sampleRate
simulationTime = 5.0
print('Sample Time: ', sampleTime)

# Additional parameters
counter = 0

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## QCar Initialization
myCar = QCar()

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Main Loop
try:
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = time.time()

        # Basic IO - write motor commands
        mtr_cmd = np.array([0.1*np.sin(elapsed_time()*2*np.pi/5), 0.5*np.sin(elapsed_time()*2*np.pi/2.5)])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        if mtr_cmd[1] > 0.3:
            LEDs[0] = 1
            LEDs[2] = 1
        elif mtr_cmd[1] < -0.3:
            LEDs[1] = 1
            LEDs[3] = 1
        if mtr_cmd[0] < 0:
            LEDs[5] = 1
        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)        

        # End timing this iteration
        end = time.time()

        # Calculate computation time, and the time that the thread should pause/sleep for
        computation_time = end - start
        sleep_time = sampleTime - computation_time%sampleTime

        # Pause/sleep and print out the current timestamp
        time.sleep(sleep_time)
        # print('Simulation Timestamp :', elapsed_time(), ' s, battery is at :', 100 - (12.6 - batteryVoltage)*100/(2.1), ' %, Motor Cmd : ', mtr_cmd[0], ' and steering : ', mtr_cmd[1])
        print("Simulation Timestamp : {0:5.3f}s, remaining battery capacity is at : {1:4.2f}%, motor throttle is : {2:4.2f}% PWM and the steering is : {3:3.2f} rad".format(elapsed_time(), 100 - (12.6 - batteryVoltage)*100/(2.1), mtr_cmd[0], mtr_cmd[1]))
        counter += 1

except KeyboardInterrupt:
    print("User interrupted!")

finally:    
    myCar.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
