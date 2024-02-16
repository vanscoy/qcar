# Copied From https://github.com/toolbuddy/2D-Grid-SLAM/blob/master/utils.py
import numpy as np
import cv2
from math import *

def EndPoint(robot_pos, angles, dists):
    pts_list = []
    for i in range(len(angles)):
        theta = robot_pos[2] + angles[i]
        pts_list.append([robot_pos[0] + dists[i] * np.cos(theta), robot_pos[1] + dists[i] * np.sin(theta)])
    return pts_list

def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def Bresenham(x0, x1, y0, y1):
    rec = []
    "Bresenham's line algorithm"
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    #print(rec)
    return rec

def Image2Map(fname):
        im = cv2.imread(fname)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m

def Map2Image(m):
    img = (255*m).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def Rotation2Deg(R):
    cos = R[0,0]
    sin = R[1,0]
    theta = np.rad2deg(np.arccos(np.abs(cos)))
    
    if cos>0 and sin>0:
        return theta
    elif cos<0 and sin>0:
        return 180-theta
    elif cos<0 and sin<0:
        return 180+theta
    elif cos>0 and sin<0:
        return 360-theta
    elif cos==0 and sin>0:
        return 90.0
    elif cos==0 and sin<0:
        return 270.0
    elif cos>0 and sin==0:
        return 0.0
    elif cos<0 and sin==0:
        return 180.0

def xPos(heading, dist=0.0):
    return dist * np.cos(heading)

def yPos(heading, dist=0.0):
    return dist * np.sin(heading)

# .26 is the length from axle to axle in m
def anglePos(theta, phi, dist=0.0):
    radius = .26 / np.tan(phi)
    angle = dist/radius
    return  theta + angle



def posUpdate(robot_pos, turnAngle, dist=0.0):
    xNew, yNew = xPos(robot_pos[2], dist), yPos(robot_pos[2], dist)
    
    thetaNew = anglePos(robot_pos[2], turnAngle, dist)
    
    return [xNew, yNew, thetaNew]


