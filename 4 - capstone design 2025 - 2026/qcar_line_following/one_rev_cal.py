from Quanser.product_QCar import QCar
import numpy as np
import time

car = QCar()
LEDs = np.array([0,0,0,0,0,0,0,0])

def get_enc(ret):
    try:
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
    # Ensure motors off
    car.read_write_std(np.array([0.0,0.0]), LEDs)
    input("Ready. Make sure wheel is free to rotate. Press Enter, then rotate ONE full wheel revolution, then press Enter again.")
    ret0 = car.read_write_std(np.array([0.0,0.0]), LEDs)
    enc0 = get_enc(ret0)
    print("Before:", enc0)
    input("Now rotate the wheel exactly one revolution (physically) and press Enter.")
    ret1 = car.read_write_std(np.array([0.0,0.0]), LEDs)
    enc1 = get_enc(ret1)
    print("After:", enc1)
    if enc0 is None or enc1 is None:
        print("Could not read encoder value (None). Check connection/API).")
    else:
        delta = enc1 - enc0
        print(f'Counts observed for 1 wheel rev: {delta}')
finally:
    try:
        car.read_write_std(np.array([0.0,0.0]), LEDs)
    except Exception:
        pass
    car.terminate()