from Quanser.q_essential import LIDAR
import time

lidar = LIDAR(num_measurements=360)

try:
    for _ in range(20):
        lidar.read()
        print("Distances:", lidar.distances[:5])
        print("Angles:   ", lidar.angles[:5])
        time.sleep(0.1)
finally:
    lidar.terminate()