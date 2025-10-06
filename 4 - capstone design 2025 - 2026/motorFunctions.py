"""
Motor helper functions and a small CLI for the QCar.

This module provides simple clamping helpers and extraction helpers for the
project convention where motor command sequences are [throttle, steering]
with throttle at index 0 and steering at index 1. The interactive CLI (run
from the terminal) lets the user set throttle and steering and query the
robot for status using the same read/write call used in `hardwareTest.py`.
"""
from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D, LIDAR
# from speedCalc import *
import time
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import cv2

from typing import Sequence

myCar = QCar()

motor = np.array([0, 0])

def setThrottle(value: float):
    motor[0] = value/4
    mtr_cmd = motor
    
# Accepts Steering Range of ±28 degrees
def setSteering(value: float):
    motor[1] = value/28/4
    mtr_cmd = motor


def getThrottle() -> float:
    return motor[0]


def getSteering(mtr_cmd: Sequence[float]) -> float:
    return motor[1]

def qcar_cli():
    """Interactive CLI to set throttle and steering (deg) for the QCar."""
    print("QCar Control CLI")
    print("================")
    print("Commands:")
    print("  b  set both throttle and steering")
    print("  t  set throttle (velocity command)")
    print("  a  set steering (angle in degrees, ±28 max)")
    print("  p  print current command")
    print("  s  send command to QCar and read status")
    print("  q  quit\n")

    leds = [0]*8  # base LED pattern

    while True:
        cmd = input("Enter command: ").strip().lower()
        if cmd == 'q':
            print("Exiting CLI.")
            break
        elif cmd == 'b':
            try:
                th = float(input("Throttle (-1.0..1.0): "))
                st = float(input("Steering angle (°, -28..28): "))
            except ValueError:
                print("Invalid numeric input.")
                continue
            setThrottle(th)
            setSteering(st)
            print(f"Set: Throttle={getThrottle():.2f}, Steering={getSteering(motor)*28:.2f}°")
        elif cmd == 't':
            try:
                th = float(input("Throttle (-1.0..1.0): "))
            except ValueError:
                print("Invalid numeric input.")
                continue
            setThrottle(th)
            print(f"Throttle set to {getThrottle():.2f}")
        elif cmd == 'a':
            try:
                st = float(input("Steering angle (°, -28..28): "))
            except ValueError:
                print("Invalid numeric input.")
                continue
            setSteering(st)
            print(f"Steering set to {getSteering(motor)*28:.2f}°")
        elif cmd == 'p':
            print(f"Throttle={getThrottle():.2f}, Steering={getSteering(motor)*28:.2f}°")
        elif cmd == 's':
            try:
                current, batteryVoltage, encoderCounts = myCar.read_write_std(motor, leds)
                battery_pct = max(0, min(100, 100 - (12.6 - batteryVoltage) * 100 / 2.1))
                print(f"Motor={motor}, Current={current:.2f}, Battery={batteryVoltage:.2f} V ({battery_pct:.1f}%), Encoders={encoderCounts}")
            except Exception as exc:
                print(f"Error communicating with QCar: {exc}")
        else:
            print("Unknown command")

        time.sleep(0.05)

if __name__ == "__main__":
    qcar_cli()