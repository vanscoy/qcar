# Copied from https://github.com/toolbuddy/2D-Grid-SLAM/blob/master/GridMap.py
import numpy as np
import utils

# Map making class
# Uses Log odds form for probabilities
class GridMap:
    # gmap saves into a dictionary of grid cels {x,y} and probabilities with max/min set by map_param[3/4]
    def __init__(self, map_param, gsize=1.0):
        self.map_param = map_param
        self.gmap = {}
        self.gsize = gsize
        self.center = np.array([0,0])
        self.boundary = [9999, -9999, 9999, -9999] 

    # Returns the probability of an object in that grid cell
    # Returns .5 when not initialized
    def GetGridProb(self, pos):
        if pos in self.gmap:
            return np.exp(self.gmap[pos]) / (1.0 + np.exp(self.gmap[pos]))
        else:
            return 0.5

    # Converts a coordinate to grid cell and calls GetGridProb
    def GetCoordProb(self, pos):
        x, y = int(round(pos[0]/self.gsize)), int(round(pos[1]/self.gsize))
        return self.GetGridProb((x,y))

    # Given a range of x and y values, map that range to an array 
    def GetMapProb(self, x0, x1, y0, y1):
        map_prob = np.zeros((y1-y0, x1-x0))
        idx = 0
        for i in range(x0, x1):
            idy = 0
            for j in range(y0, y1):
                map_prob[idy, idx] = self.GetGridProb((i,j))
                if i ==0 and j == 0:
                    self.center = np.array([idx, idy])
                idy += 1
            idx += 1
        return map_prob

    # Given a start point and an end point, draw a line using Bresenhams line algorithm and update each grid cell
    # The grid cell where the endpoint is is where an object is expected to be, and the other point on that line are expected to be empty
    def GridMapLine(self, x0, x1, y0, y1):
        
        # Scale the position
        x0, x1 = int(round(x0/self.gsize)), int(round(x1/self.gsize))
        y0, y1 = int(round(y0/self.gsize)), int(round(y1/self.gsize))

        rec = utils.Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            if i < len(rec)-2:
                change = self.map_param[0]
            else:
                change = self.map_param[1]

            if rec[i] in self.gmap:
                self.gmap[rec[i]] += change
            else:
                self.gmap[rec[i]] = change
                if rec[i][0] < self.boundary[0]:
                    self.boundary[0] = rec[i][0]
                elif rec[i][0] > self.boundary[1]:
                    self.boundary[1] = rec[i][0]                  
                if rec[i][1] < self.boundary[2]:
                    self.boundary[2] = rec[i][1]
                elif rec[i][1] > self.boundary[3]:
                    self.boundary[3] = rec[i][1]


            if self.gmap[rec[i]] > self.map_param[2]:
                self.gmap[rec[i]] = self.map_param[2]
            if self.gmap[rec[i]] < self.map_param[3]:
                self.gmap[rec[i]] = self.map_param[3]

"""
    def EmptyMapLine(self, x0, x1, y0, y1):
        # Scale the position
        #print(y1)
        x0, x1 = int(round(x0/self.gsize)), int(round(x1/self.gsize))
        y0, y1 = int(round(y0/self.gsize)), int(round(y1/self.gsize))
        #print(y1)

        rec = utils.Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
    

            if rec[i] in self.gmap:
                self.gmap[rec[i]] += self.map_param[0]
            else:
                self.gmap[rec[i]] = self.map_param[0]
                if rec[i][0] < self.boundary[0]:
                    self.boundary[0] = rec[i][0]
                    #print("Boundary Changed 1")
                elif rec[i][0] > self.boundary[1]:
                    self.boundary[1] = rec[i][0]
                    #print("Boundary Changed 2")                    
                if rec[i][1] < self.boundary[2]:
                    self.boundary[2] = rec[i][1]
                    #print("Boundary Changed 3")
                elif rec[i][1] > self.boundary[3]:
                    self.boundary[3] = rec[i][1]
                    #print("Boundary Changed 4")


            if self.gmap[rec[i]] > self.map_param[2]:
                self.gmap[rec[i]] = self.map_param[2]
            if self.gmap[rec[i]] < self.map_param[3]:
                self.gmap[rec[i]] = self.map_param[3]
"""
    
if __name__ == '__main__':
    #lo_occ, lo_free, lo_max, lo_min
    map_param = [0.9, -0.7, 5.0, -5.0]
    m = GridMap(map_param)
    pos = (0.0,0.0)
    m.gmap[pos] = 0.1
    print(m.GetProb(pos))
    print(m.GetProb((0,0)))
