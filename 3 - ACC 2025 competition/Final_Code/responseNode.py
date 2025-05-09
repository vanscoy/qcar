import math
import time

class PathTracker:
    def getHeading(self, speed, turnRadius, turnTime, currentHeading):
        headingChange = 0
        turnTime = time.time() - turnTime
        #print(speed)
        #print(turnRadius)
        #print(turnTime)
        if turnRadius > 0:
            headingChange = round((speed/turnRadius)*turnTime, 6)
            currentHeading += headingChange
        elif turnRadius < 0:
            headingChange = round((speed/turnRadius)*turnTime, 6)
            currentHeading -= headingChange
        print(f'Change in Heading: {headingChange}')
        
        if currentHeading > 360:
            return round(currentHeading - 360, 5)
        elif currentHeading < 0:
            return round(currentHeading + 360, 5)
        else:
            return round(currentHeading, 5)
    
    def findPos(self, currentPos, currentHeading, speed):
        if currentHeading >= 0 and currentHeading < 90:
            currentPos[0] += speed*math.cos(currentHeading)
            currentPos[1] += speed*math.sin(currentHeading)
        elif currentHeading >= 90 and currentHeading < 180:
            currentPos[0] -= speed*math.cos(currentHeading)
            currentPos[1] += speed*math.sin(currentHeading)
        elif currentHeading >= 180 and currentHeading < 270:
            currentPos[0] -= speed*math.cos(currentHeading)
            currentPos[1] -= speed*math.sin(currentHeading)
        elif currentHeading >= 270 and currentHeading < 360:
            currentPos[0] += speed*math.cos(currentHeading)
            currentPos[1] -= speed*math.sin(currentHeading)
        currentPos[0], currentPos[1] = round(currentPos[0], 2), round(currentPos[1], 2)
        return currentPos

    def turnCar(self, angle, speed, turn):
        if turn:
            angle = .5 # Placeholder value
            speed = .5 # Placeholder value
            time.sleep(5) # Placeholder time to turn
    
    def calcNodeDistance(self, currentPos, goalPos):
        return math.sqrt(((goalPos[0] - currentPos[1])^2)+((goalPos[0] - currentPos[1])^2))
    
    def detectPath(self, currentPos, goalPos):
        distance = self.calcNodeDistance(currentPos, goalPos)
        if distance < 1:
            return True
        else:
            return False