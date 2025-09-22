from Quanser.product_QCar import QCar
import time
import numpy as np

# Timing Parameters
SAMPLE_RATE = 1000
SAMPLE_TIME = 1 / SAMPLE_RATE
SIMULATION_TIME = 5.0

def elapsed_time(start_time):
    return time.time() - start_time

def calculate_leds(mtr_cmd):
    leds = np.zeros(8, dtype=int)
    leds[6:8] = 1
    if mtr_cmd[1] > 0.3:
        leds[[0, 2]] = 1
    elif mtr_cmd[1] < -0.3:
        leds[[1, 3]] = 1
    if mtr_cmd[0] < 0:
        leds[5] = 1
    return leds

def main():
    print(f'Sample Time: {SAMPLE_TIME}')
    myCar = QCar()
    start_time = time.time()
    counter = 0

    try:
        while elapsed_time(start_time) < SIMULATION_TIME:
            loop_start = time.time()
            t = elapsed_time(start_time)

            # Motor commands
            mtr_cmd = np.array([
                0.1 * np.sin(t * 2 * np.pi / 5),
                0.5 * np.sin(t * 2 * np.pi / 2.5)
            ])
            leds = calculate_leds(mtr_cmd)

            current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, leds)

            # Print status
            battery_pct = 100 - (12.6 - batteryVoltage) * 100 / 2.1
            print(f"Simulation Timestamp : {t:5.3f}s, remaining battery capacity is at : {battery_pct:4.2f}%, "
                  f"motor throttle is : {mtr_cmd[0]:4.2f}% PWM and the steering is : {mtr_cmd[1]:3.2f} rad")

            # Sleep to maintain sample rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, SAMPLE_TIME - elapsed)
            time.sleep(sleep_time)
            counter += 1

    except KeyboardInterrupt:
        print("User interrupted!")

    finally:
        myCar.terminate()

if __name__ == "__main__":
    main()
