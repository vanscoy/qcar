from Quanser.product_QCar import QCar
import numpy as np
import time

car = QCar()
LEDs = np.array([0,0,0,0,0,0,0,0])
sample_dt = 0.05

def run_phase(cmd, duration=1.0):
    print(f'Running cmd={cmd} for {duration}s')
    start=time.time()
    while time.time()-start < duration:
        try:
            ret = car.read_write_std(np.array(cmd), LEDs)
            print(f'{time.time():.3f}: ret -> {repr(ret)}')
        except Exception as e:
            print('read_write_std exception', e)
        time.sleep(sample_dt)
    # stop
    try:
        car.read_write_std(np.array([0.0,0.0]), LEDs)
    except Exception:
        pass

# Phase A: excite first slot
run_phase([0.06, 0.0], duration=1.0)
time.sleep(0.2)

# Phase B: excite second slot
run_phase([0.0, 0.06], duration=1.0)
time.sleep(0.2)

car.terminate()
print('Done')