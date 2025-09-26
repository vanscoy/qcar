"""
Motor helper functions and a small CLI for the QCar.

This module provides simple clamping helpers and extraction helpers for the
project convention where motor command sequences are [throttle, steering]
with throttle at index 0 and steering at index 1. The interactive CLI (run
from the terminal) lets the user set throttle and steering and query the
robot for status using the same read/write call used in `hardwareTest.py`.
"""
from typing import Sequence

# Allowed ranges.
THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0
STEERING_MIN = -1.0
STEERING_MAX = 1.0


def setThrottle(value: float) -> float:
    """Clamp and return the throttle command within [-1.0, 1.0].

    Accepts numeric types. Returns a float in the allowed range.
    """
    try:
        v = float(value)
    except Exception as exc:
        raise TypeError(f"throttle value must be numeric: {exc}")
    return max(THROTTLE_MIN, min(THROTTLE_MAX, v))


def setSteering(value: float) -> float:
    """Clamp and return the steering command within [-1.0, 1.0].

    Accepts numeric types. Returns a float in the allowed range.
    """
    try:
        v = float(value)
    except Exception as exc:
        raise TypeError(f"steering value must be numeric: {exc}")
    return max(STEERING_MIN, min(STEERING_MAX, v))


def getThrottle(mtr_cmd: Sequence[float]) -> float:
    """Extract throttle value from motor command sequence (index 0)."""
    if mtr_cmd is None:
        raise ValueError("mtr_cmd is None")
    try:
        return float(mtr_cmd[0])
    except Exception:
        raise ValueError("mtr_cmd must be an indexable sequence with throttle at index 0")


def getSteering(mtr_cmd: Sequence[float]) -> float:
    """Extract steering value from motor command sequence (index 1)."""
    if mtr_cmd is None:
        raise ValueError("mtr_cmd is None")
    try:
        return float(mtr_cmd[1])
    except Exception:
        raise ValueError("mtr_cmd must be an indexable sequence with steering at index 1")


def calculate_leds(mtr_cmd: Sequence[float]):
    """Return an 8-element LED array matching the logic used in hardwareTest.py.

    The returned object is a list of ints (0/1) representing LED on/off.
    """
    leds = [0] * 8
    leds[6] = 1
    leds[7] = 1
    try:
        if getSteering(mtr_cmd) > 0.3:
            leds[0] = 1
            leds[2] = 1
        elif getSteering(mtr_cmd) < -0.3:
            leds[1] = 1
            leds[3] = 1
        if getThrottle(mtr_cmd) < 0:
            leds[5] = 1
    except Exception:
        # If the provided mtr_cmd isn't valid, just return the base LED pattern
        pass
    return leds


def _interactive_loop():
    """Run a simple terminal interface to set/get throttle and steering.

    This function instantiates the QCar and performs read/write operations
    only when run as a script (not on import).
    """
    from Quanser.product_QCar import QCar
    import time

    qcar = QCar()
    mtr_cmd = [0.0, 0.0]

    try:
        while True:
            print("\nCommands:")
            print("  b  set both throttle and steering")
            print("  t  set throttle")
            print("  a  set steering")
            print("  s  send current command to car and read status")
            print("  p  print current command")
            print("  q  quit")
            cmd = input("Enter command: ").strip().lower()

            if cmd == 'q':
                break
            elif cmd == 'b':
                try:
                    th = float(input("Throttle (-1.0..1.0): "))
                    st = float(input("Steering (-1.0..1.0): "))
                except ValueError:
                    print("Invalid numeric input")
                    continue
                mtr_cmd[0] = setThrottle(th)
                mtr_cmd[1] = setSteering(st)
                print(f"Set mtr_cmd = {mtr_cmd}")
            elif cmd == 't':
                try:
                    th = float(input("Throttle (-1.0..1.0): "))
                except ValueError:
                    print("Invalid numeric input")
                    continue
                mtr_cmd[0] = setThrottle(th)
                print(f"Throttle set to {mtr_cmd[0]}")
            elif cmd == 'a':
                try:
                    st = float(input("Steering (-1.0..1.0): "))
                except ValueError:
                    print("Invalid numeric input")
                    continue
                mtr_cmd[1] = setSteering(st)
                print(f"Steering set to {mtr_cmd[1]}")
            elif cmd == 'p':
                print(f"Current command: throttle={getThrottle(mtr_cmd)}, steering={getSteering(mtr_cmd)}")
            elif cmd == 's':
                # send command and read status via read_write_std (same as hardwareTest)
                leds = calculate_leds(mtr_cmd)
                try:
                    current, batteryVoltage, encoderCounts = qcar.read_write_std(mtr_cmd, leds)
                    battery_pct = 100 - (12.6 - batteryVoltage) * 100 / 2.1
                    print(f"Current: {current}, Battery V: {batteryVoltage}, Battery %: {battery_pct:.2f}%, Encoders: {encoderCounts}")
                except Exception as exc:
                    print(f"Error communicating with QCar: {exc}")
            else:
                print("Unknown command")

            # small pause to avoid busy loop
            time.sleep(0.05)

    finally:
        # make a best-effort cleanup
        if hasattr(qcar, 'terminate'):
            try:
                qcar.terminate()
            except Exception:
                pass


if __name__ == '__main__':
    _interactive_loop()
