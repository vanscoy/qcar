from Quanser.product_QCar import QCar
import numpy as np
import time

ENC_COUNTS_PER_REV_GUESS = 2880  # internal fallback
WHEEL_RADIUS_M = 0.033

car = QCar()
LEDs = np.array([0,0,0,0,0,0,0,0])

def read_enc():
    try:
        ret = car.read_write_std(np.array([0.0,0.0]), LEDs)
        if isinstance(ret, (list, tuple)) and len(ret) >= 3:
            a = ret[2]
            if isinstance(a, (list, tuple, np.ndarray)):
                if len(a) >= 1:
                    return int(a[0])
            else:
                return int(a)
    except Exception:
        pass
    try:
        fn = getattr(car, 'read_encoder', None)
        if callable(fn):
            v = fn()
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 1:
                return int(v[0])
            if isinstance(v, int):
                return int(v)
    except Exception:
        pass
    return None

try:
    input("Elevate car so wheels spin freely and mark one wheel visibly. Press Enter to start.")
    enc0 = read_enc()
    print("Encoder before:", enc0)
    # run motor briefly
    cmd = np.array([0.06, 0.0])
    run_t = 2.0
    start=time.time()
    while time.time()-start < run_t:
        car.read_write_std(cmd, LEDs)
        time.sleep(0.05)
    car.read_write_std(np.array([0.0,0.0]), LEDs)
    enc1 = read_enc()
    print("Encoder after:", enc1)
    if enc0 is None or enc1 is None:
        print("Could not read enc values.")
    else:
        delta = enc1 - enc0
        print(f'Raw delta in {run_t}s: {delta}')
        print("Now visually count how many full wheel revolutions occurred (or estimate).")
        n = float(input("Enter observed revolutions (e.g., 3): "))
        if n > 0:
            measured = delta / n
            print(f'Estimated counts_per_wheel_rev = {measured:.2f}')
            print('You can set ENC_COUNTS_PER_REV in the main script to this value for correct conversions.')
        else:
            print('No valid revolution count entered.')
finally:
    try:
        car.read_write_std(np.array([0.0,0.0]), LEDs)
    except Exception:
        pass
    car.terminate()
    print('Done')