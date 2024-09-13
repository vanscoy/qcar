from Quanser.product_QCar import QCar
import time
import numpy as np
import utils

# Helper object to find speed and distance of a given qcar
class speedCalc:
    def __init__(self, robot_pos, qCar, t = 0.0):
        self.robot_pos = robot_pos
        self.qCar = qCar
        self.begin_encoder = self.qCar.read_encoder()
        self.t = t

    # Helper function to find the time since object initialization
    def elapsed_time(self):
        return time.time() - self.t
    
    # Returns the speed (m/s) found in between function calls
    def encoder_speed(self):
        totalTime = time.time() - self.t
        self.t = time.time()
        currentEncoder = self.qCar.read_encoder()
        encoderChange = currentEncoder - self.begin_encoder
        self.begin_encoder = currentEncoder
        
        return ((encoderChange/31844) * (.066 * np.pi)) / totalTime

    # Returns the distance (m) traveled in between function calls
    def encoder_dist(self):
        currentEncoder = self.qCar.read_encoder()
        encoderChange = currentEncoder - self.begin_encoder
        self.begin_encoder = currentEncoder
        return (encoderChange/31844) * (.066 * np.pi)

    
