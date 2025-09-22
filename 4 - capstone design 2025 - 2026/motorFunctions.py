from Quanser.product_QCar import QCar
from typing import Sequence

# Allowed ranges.
THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0
STEERING_MIN = -1.0
STEERING_MAX = 1.0


def setThrottle(value: float) -> float:
    """Clamp and return the throttle command within [-1.0, 1.0]."""
    return max(THROTTLE_MIN, min(THROTTLE_MAX, value))


def setSteering(value: float) -> float:
    """Clamp and return the steering command within [-1.0, 1.0]."""
    return max(STEERING_MIN, min(STEERING_MAX, value))


def getThrottle(mtr_cmd: Sequence[float]) -> float:
    """Extract throttle value from motor command sequence."""
    return float(mtr_cmd[0])


def getSteering(mtr_cmd: Sequence[float]) -> float:
    """Extract steering value from motor command sequence."""
    return float(mtr_cmd[1])

if __name__ == "__main__":
    with QCar() as qcar:
        while True:
            # Take user input
            throttle_in = float(input("Enter throttle value (-1.0 to 1.0): "))
            steering_in = float(input("Enter steering value (-1.0 to 1.0): "))

            # Clamp using helpers
            throttle = setThrottle(throttle_in)
            steering = setSteering(steering_in)

            # Build motor command
            mtr_cmd = [throttle, steering]
            qcar.set_cmd(mtr_cmd)

            # Print results
            print(f"Clamped Throttle: {getThrottle(mtr_cmd)}")
            print(f"Clamped Steering: {getSteering(mtr_cmd)}")
            print()
