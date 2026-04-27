# Setup_Real_Scenario_scaled10.py
import os
import time

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.qcar_flooring import QLabsQCarFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.yield_sign import QLabsYieldSign
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight


# ============================
# GLOBAL SCALE KNOB
# ============================
WORLD = 10.0   # 10× positions AND 10× object sizes


def main():
    os.system("cls")

    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost")
        print("Connected to QLabs")
    except:
        print("Unable to connect to QLabs")
        return

    # Clean slate
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # Build environment
    setup(
        qlabs=qlabs,
        initialPosition=[-1.205, -0.83, 0.005],
        initialOrientation=[0, 0, -44.7],
    )

    # ============================
    # TRAFFIC LIGHTS
    # ============================
    trafficLight1 = QLabsTrafficLight(qlabs)
    trafficLight2 = QLabsTrafficLight(qlabs)
    trafficLight3 = QLabsTrafficLight(qlabs)
    trafficLight4 = QLabsTrafficLight(qlabs)

    trafficLight1.spawn_id_degrees(1, P([0.6,  1.55, 0.006]), [0, 0,   0], SC([0.1]*3), 0, False)
    trafficLight2.spawn_id_degrees(2, P([-0.6, 1.28, 0.006]), [0, 0,  90], SC([0.1]*3), 0, False)
    trafficLight3.spawn_id_degrees(3, P([-0.37,0.30, 0.006]), [0, 0, 180], SC([0.1]*3), 0, False)
    trafficLight4.spawn_id_degrees(4, P([0.75, 0.48, 0.006]), [0, 0, -90], SC([0.1]*3), 0, False)

    print("Starting Traffic Light Sequence")
    flag = 0

    while True:
        if flag == 0:
            trafficLight1.set_color(QLabsTrafficLight.COLOR_RED)
            trafficLight3.set_color(QLabsTrafficLight.COLOR_RED)
            trafficLight2.set_color(QLabsTrafficLight.COLOR_GREEN)
            trafficLight4.set_color(QLabsTrafficLight.COLOR_GREEN)
        elif flag == 1:
            trafficLight2.set_color(QLabsTrafficLight.COLOR_YELLOW)
            trafficLight4.set_color(QLabsTrafficLight.COLOR_YELLOW)
        elif flag == 2:
            trafficLight1.set_color(QLabsTrafficLight.COLOR_GREEN)
            trafficLight3.set_color(QLabsTrafficLight.COLOR_GREEN)
            trafficLight2.set_color(QLabsTrafficLight.COLOR_RED)
            trafficLight4.set_color(QLabsTrafficLight.COLOR_RED)
        else:
            trafficLight1.set_color(QLabsTrafficLight.COLOR_YELLOW)
            trafficLight3.set_color(QLabsTrafficLight.COLOR_YELLOW)

        flag = (flag + 1) % 4
        time.sleep(5)


def setup(qlabs, initialPosition, initialOrientation):
    # ============================
    # OFFSETS (SCALED!)
    # ============================
    x_offset = 0
    y_offset = 0

    # -------- helpers ----------
    global P, SC

    def P(p):
        return [p[0] * WORLD + x_offset,
                p[1] * WORLD + y_offset,
                p[2] * WORLD]

    def SC(s):
        return [s[0] * WORLD,
                s[1] * WORLD,
                s[2] * WORLD]

    # ============================
    # SYSTEM / FLOOR
    # ============================
    QLabsSystem(qlabs).set_title_string(
        "ACC Self Driving Car Competition", True
    )

    QLabsQCarFlooring(qlabs).spawn_degrees(
        [x_offset, y_offset, 0.006 * WORLD], [0, 0, -90]
    )

    # ============================
    # WALLS
    # ============================
    wall = QLabsWalls(qlabs)
    wall.set_enable_dynamics(False)

    for y in range(5):
        wall.spawn_degrees(P([-2.4, -y + 2.55, 0.006]), [0, 0, 0])
    for x in range(5):
        wall.spawn_degrees(P([-1.9 + x, 3.05, 0.006]), [0, 0, 90])
    for y in range(6):
        wall.spawn_degrees(P([2.4, -y + 2.55, 0.006]), [0, 0, 0])
    for x in range(4):
        wall.spawn_degrees(P([-0.9 + x, -3.05, 0.006]), [0, 0, 90])

    wall.spawn_degrees(P([-2.03, -2.275, 0.006]), [0, 0, 48])
    wall.spawn_degrees(P([-1.575, -2.7, 0.006]), [0, 0, 48])

    # ============================
    # QCAR
    # ============================
    car = QLabsQCar2(qlabs)
    car.spawn_id(
        0, P(initialPosition), initialOrientation,
        SC([0.1, 0.1, 0.1]), 0, True
    )

    # ============================
    # CAMERAS
    # ============================
    QLabsFreeCamera(qlabs).spawn_degrees(P([0.15, 1.7, 5.0]), [0, 90, 0])
    cam = QLabsFreeCamera(qlabs)
    cam.spawn_degrees(P([-0.36, -3.691, 2.652]), [0, 47, 90])
    cam.possess()

    # ============================
    # SIGNS / CROSSWALKS / LINES
    # ============================
    for cls, data in [
        (QLabsStopSign, [
            ([-1.5, 3.6, 0.006], -35),
            ([-1.5, 2.2, 0.006],  35),
            ([2.41, 0.206,0.006],-90),
            ([1.766,1.697,0.006], 90)
        ]),
        (QLabsYieldSign, [
            ([0.0,-1.3,0.006],-180),
            ([2.4, 3.2,0.006], -90),
            ([1.1, 2.8,0.006],-145),
            ([0.49,3.8,0.006], 135)
        ]),
    ]:
        obj = cls(qlabs)
        for pos, rot in data:
            obj.spawn_degrees(P(pos), [0,0,rot], SC([0.1]*3), False)

    cw = QLabsCrosswalk(qlabs)
    for pos, rot in [
        ([-1.87,0.195,0.006],0), ([-0.5,0.95,0.006],90),
        ([0.15,0.32,0.006],0), ([0.75,0.95,0.006],90),
        ([0.13,1.57,0.006],0), ([1.45,0.95,0.006],90)
    ]:
        cw.spawn_degrees(P(pos), [0,0,rot], SC([0.1,0.1,0.075]), 0)

    line = QLabsBasicShape(qlabs)
    for pos, rot, sc in [
        ([2.21,0.2,0.006],0,[0.27,0.02,0.006]),
        ([1.951,1.68,0.006],0,[0.27,0.02,0.006]),
        ([-0.05,-1.02,0.006],90,[0.38,0.02,0.006])
    ]:
        line.spawn_degrees(P(pos), [0,0,rot], SC(sc), False)

    # ============================
    # START REAL-TIME MODEL
    # ============================
    rt = os.path.join(os.environ["RTMODELS_DIR"], "QCar2/QCar2_Workspace_studio")
    QLabsRealTime().start_real_time_model(os.path.normpath(rt))


if __name__ == "__main__":
    main()