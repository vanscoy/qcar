import heapq
import random
from Quanser.q_ui import gamepadViaTarget

class Cell:
    def __init__(self, name, x, y, reachable, special):
        self.name = name
        self.x, self.y = x, y
        self.special = special
        self.reachable = reachable
        self.g = self.h = self.f = float('inf')
        self.parent = None

    def __lt__(self, other):
        return (self.f + self.h) < (other.f + other.h)

class Astar:
    def __init__(self, openList, closedList, cells):
        self.openList, self.closedList = openList, closedList
        self.cells = cells

    def getHueristic(self, cell, goal):
            return abs(cell.x - goal.x) + abs(cell.y - goal.y)
        
    def getNeighbors(self, cell):
            neighbors = []
            for x in cell.reachable:
                    neighbors.append(self.cells[x-1])
            return neighbors
        
    def updateCells(self, neighbor, cell):
            neighbor.g = cell.g + 1
            neighbor.h = self.getHueristic(neighbor, self.goal)
            neighbor.f, neighbor.parent = neighbor.g + neighbor.h, cell
        
    def search(self, start, goal):
        self.start, self.goal = start, goal
        heapq.heappush(self.openList, (self.start.f, self.start))
        while self.openList:
            f, cell = heapq.heappop(self.openList)
            self.closedList.append(cell)
            if cell == self.goal:
                path = []
                while cell.parent:
                    path.append((cell.name))
                    cell = cell.parent
                path.append((self.start.name))
                path.reverse()
                return path
            neighbors = self.getNeighbors(cell)
            for neighbor in neighbors:
                if neighbor in self.closedList or not neighbor.reachable:
                    continue
                if neighbor not in self.openList or neighbor.g > cell.g + 1:
                    self.updateCells(neighbor, cell)
                    if neighbor not in self.openList:
                        heapq.heappush(self.openList, (neighbor.f, neighbor))

def checkPath(start, goal, pathing):
    path = []
    if start.name == 1 and goal.name == 12:
        path = [1,2,3,4,5,22,12]
    if start.name == 1 and goal.name == 22:
        path = [1,2,3,4,5,22]
    if start.name == 2 and goal.name == 12:
        path = [2,3,4,5,22,12]
    if start.name == 2 and goal.name == 22:
        path = [2,3,4,5,22]
    if start.name == 3 and goal.name == 12:
        path = [3,4,5,22,12]
    if start.name == 3 and goal.name == 22:
        path = [3,4,5,22]
    if start.name == 4 and goal.name == 18:
        path = [4,5,6,7,8,18]
    if start.name == 5 and goal.name == 4:
        path = [5,22,12,4]
    if start.name == 5 and goal.name == 18:
        path = [5,6,7,8,18]
    if start.name == 6 and goal.name == 4:
        path = [6,7,8,12,4]
    if start.name == 6 and goal.name == 5:
        path = [6,7,8,12,4,5]
    if start.name == 6 and goal.name == 15:
        path = [6,7,8,18,19,15]
    if start.name == 6 and goal.name == 18:
        path = [6,7,8,18]
    if start.name == 6 and goal.name == 19:
        path = [6,7,8,18,19]
    if start.name == 6 and goal.name == 22:
        path = [6,7,8,12,4,5,22]
    if start.name == 7 and goal.name == 4:
        path = [7,8,12,4]
    if start.name == 7 and goal.name == 5:
        path = [7,8,12,4,5]
    if start.name == 7 and goal.name == 6:
        path = [7,8,18,6]
    if start.name == 7 and goal.name == 15:
        path = [7,8,18,19,15]
    if start.name == 7 and goal.name == 18:
        path = [7,8,18]
    if start.name == 7 and goal.name == 19:
        path = [7,8,18,19]
    if start.name == 7 and goal.name == 22:
        path = [7,8,12,4,5,22]
    if start.name == 8 and goal.name == 4:
        path = [8,12,4]
    if start.name == 8 and goal.name == 5:
        path = [8,12,4,5]
    if start.name == 8 and goal.name == 10:
        path = [8,12,13,10]
    if start.name == 8 and goal.name == 12:
        path = [8,12]
    if start.name == 8 and goal.name == 13:
        path = [8,12,13]
    if start.name == 8 and goal.name == 22:
        path = [8,12,4,5,22]
    if start.name == 9 and goal.name == 12:
        path = [9,3,4,5,22,12]
    if start.name == 9 and goal.name == 20:
        path = [9,3,16,23,20]
    if start.name == 9 and goal.name == 22:
        path = [9,3,4,5,22]
    if start.name == 9 and goal.name == 23:
        path = [9,3,16,23]
    if start.name == 10 and goal.name == 1:
        path = [10,18,6,7,1]
    if start.name == 10 and goal.name == 2:
        path = [10,18,6,7,1,2]
    if start.name == 10 and goal.name == 6:
        path = [10,18,6]
    if start.name == 10 and goal.name == 7:
        path = [10,18,6,7]
    if start.name == 10 and goal.name == 8:
        path = [10,18,6,7,8]
    if start.name == 10 and goal.name == 15:
        path = [10,18,19,15]
    if start.name == 10 and goal.name == 16:
        path = [10,18,19,15,16]
    if start.name == 10 and goal.name == 18:
        path = [10,18]
    if start.name == 10 and goal.name == 19:
        path = [10,18,19]
    if start.name == 12 and goal.name == 3:
        path = [12,4,5,22,9,3]
    if start.name == 12 and goal.name == 9:
        path = [12,4,5,22,9]
    if start.name == 12 and goal.name == 15:
        path = [12,13,10,18,19,15]
    if start.name == 12 and goal.name == 16:
        path = [12,13,10,18,19,15,16]
    if start.name == 12 and goal.name == 18:
        path = [12,13,10,18]
    if start.name == 12 and goal.name == 19:
        path = [12,13,10,18,19]
    if start.name == 12 and goal.name == 20:
        path = [12,13,10,23,20]
    if start.name == 12 and goal.name == 23:
        path = [12,13,10,23]
    if start.name == 13 and goal.name == 1:
        path = [13,10,18,6,7,1]
    if start.name == 13 and goal.name == 2:
        path = [13,10,18,6,7,1,2]
    if start.name == 13 and goal.name == 6:
        path = [13,10,18,6]
    if start.name == 13 and goal.name == 7:
        path = [13,10,18,6,7]
    if start.name == 13 and goal.name == 8:
        path = [13,10,18,6,7,8]
    if start.name == 13 and goal.name == 15:
        path = [13,10,18,19,15]
    if start.name == 13 and goal.name == 16:
        path = [13,10,18,19,15,16]
    if start.name == 13 and goal.name == 18:
        path = [13,10,18]
    if start.name == 13 and goal.name == 19:
        path = [13,10,18,19]
    if start.name == 13 and goal.name == 22:
        path = [13,10,23,20,22]
    if start.name == 15 and goal.name == 12:
        path = [15,13,10,12]
    if start.name == 15 and goal.name == 20:
        path = [15,16,23,20]
    if start.name == 15 and goal.name == 22:
        path = [15,16,23,20,22]
    if start.name == 15 and goal.name == 23:
        path = [15,16,23]
    if start.name == 16 and goal.name == 10:
        path = [16,18,19,15,13,10]
    if start.name == 16 and goal.name == 13:
        path = [16,18,19,15,13]
    if start.name == 16 and goal.name == 15:
        path = [16,18,19,15]
    if start.name == 16 and goal.name == 19:
        path = [16,18,19]
    if start.name == 18 and goal.name == 4:
        path = [18,6,7,8,12,4]
    if start.name == 18 and goal.name == 5:
        path = [18,6,7,8,12,4,5]
    if start.name == 18 and goal.name == 10:
        path = [18,19,15,13,10]
    if start.name == 18 and goal.name == 12:
        path = [18,6,7,8,12]
    if start.name == 18 and goal.name == 13:
        path = [18,19,15,13]
    if start.name == 19 and goal.name == 12:
        path = [19,15,13,10,12]
    if start.name == 19 and goal.name == 20:
        path = [19,15,16,23,20]
    if start.name == 19 and goal.name == 22:
        path = [19,15,16,23,20,22]
    if start.name == 19 and goal.name == 23:
        path = [19,15,16,23]
    if start.name == 20 and goal.name == 4:
        path = [20,22,12,4]
    if start.name == 20 and goal.name == 5:
        path = [20,22,12,4,5]
    if start.name == 20 and goal.name == 13:
        path = [20,19,15,13]
    if start.name == 20 and goal.name == 10:
        path = [20,19,15,13,10]
    if start.name == 20 and goal.name == 12:
        path = [20,22,12]
    if start.name == 20 and goal.name == 4:
        path = [20,19,15,13]
    if start.name == 22 and goal.name == 1:
        path = [22,12,4,5,6,7,1]
    if start.name == 22 and goal.name == 2:
        path = [22,12,4,5,6,7,1,2]
    if start.name == 22 and goal.name == 4:
        path = [22,12,4]
    if start.name == 22 and goal.name == 5:
        path = [22,12,4,5]
    if start.name == 22 and goal.name == 6:
        path = [22,12,4,5,6]
    if start.name == 22 and goal.name == 7:
        path = [22,12,4,5,6,7]
    if start.name == 22 and goal.name == 8:
        path = [22,12,4,5,6,7,8]
    if start.name == 22 and goal.name == 10:
        path = [22,12,13,10]
    if start.name == 22 and goal.name == 12:
        path = [22,12]
    if start.name == 22 and goal.name == 13:
        path = [22,12,13]
    if start.name == 23 and goal.name == 1:
        path = [23,20,22,12,4,5,6,7,1]
    if start.name == 23 and goal.name == 2:
        path = [23,20,22,12,4,5,6,7,1,2]
    if start.name == 23 and goal.name == 4:
        path = [23,20,22,12,4]
    if start.name == 23 and goal.name == 5:
        path = [23,20,22,12,4,5]
    if start.name == 23 and goal.name == 6:
        path = [23,20,22,12,4,5,6]
    if start.name == 23 and goal.name == 7:
        path = [23,20,22,12,4,5,6,7]
    if start.name == 23 and goal.name == 8:
        path = [23,20,22,12,4,5,6,7,8]
    if start.name == 23 and goal.name == 12:
        path = [23,20,22,12]
    if start.name == 23 and goal.name == 18:
        path = [23,20,19,15,16,18]
    if len(path) == 0:
        return pathing
    else:
        return path
        
def initCells():
    cells = []
    cells.append(Cell(1, 31, 36, [2], False))
    cells.append(Cell(2, 108, 19, [3], True))
    cells.append(Cell(3, 208, 74, [16, 4, 16], True))
    cells.append(Cell(4, 208, 168, [5], False))
    cells.append(Cell(5, 157, 230, [6, 22, 6], True))
    cells.append(Cell(6, 69, 235, [7], False))
    cells.append(Cell(7, 12, 180, [1, 8], True))
    cells.append(Cell(8, 58, 123, [9, 12, 18, 9], True))
    cells.append(Cell(9, 96, 74, [3], False))
    cells.append(Cell(10, 110, 70, [12, 18, 23, 23], True))
    cells.append(Cell(11, 144, 123, [], False))
    cells.append(Cell(12, 156, 123, [4, 13], True))
    cells.append(Cell(13, 193, 87, [10], False))
    cells.append(Cell(14, 163, 135, [], False))
    cells.append(Cell(15, 186, 180, [13, 16, 16], True))
    cells.append(Cell(16, 154, 135, [9, 18, 23, 9, 18], True))
    cells.append(Cell(17, 117, 168, [], False))
    cells.append(Cell(18, 117, 190, [6, 19, 6, 19], True))
    cells.append(Cell(19, 148, 220, [15], True))
    cells.append(Cell(20, 63, 223, [19, 22, 19, 22], True))
    cells.append(Cell(21, 101, 200, [], False))
    cells.append(Cell(22, 101, 180, [9, 12, 23, 9, 12, 23], True))
    cells.append(Cell(23, 67, 135, [20], False))
    return cells

def genPath():
    
    cells = initCells()

    while True:
        try:
            startName = int(input("Enter the start point: "))
            goalName = int(input("Enter the goal point: "))
        except ValueError:
            print("Invalid input. Please enter integers.")
            continue

        start = next((cell for cell in cells if cell.name == startName), None)
        goal = next((cell for cell in cells if cell.name == goalName), None)

        if not start or not goal:
            print("Invalid start or goal cell. Try again.")
            continue

        openList, closedList = [], []
        astar = Astar(openList, closedList, cells)
        path = astar.search(start, goal)
        path = checkPath(start, goal, path)

        if path:

            # needed path for main2 usages
            return path
        else:
            print("No path found. Try again.\n")
            
table = {12: 71,
         23: 136,
         34: 95,
         316: 83,
         45: 93,
         56: 87,
         522: 84,
         67: 86,
         78: 88,
         71: 162,
         89: 60,
         812: 87,
         818: 71,
         93: 186,
         1012: 59,
         1018: 85,
         1023: 81,
         1112: 10, 
         1213: 63,
         124: 99, 
         1310: 142, 
         1416: 11,
         1513: 89,
         1516: 58,
         1618: 56,
         169: 83,
         1623: 86,
         1718: 10,
         1819: 59,
         186: 88,
         1915: 78,
         2022: 59,
         2019: 89,
         2122: 10,
         2223: 59,
         229: 89,
         2212: 81,
         2320: 129,
         224: 83,
         924: 133,
         243: 53,
         425: 41,
         255: 52,
         1926: 43,
         2615: 35,
         628: 52,
         287: 34,
         2327: 66,
         2720: 63,}

dirTable = {12: "F",
         23: "F",
         34: "S",
         314: "L",
         45: "F",
         56: "S",
         522: "L",
         67: "F",
         78: "F",
         71: "S",
         89: "R",
         811: "S",
         817: "L",
         93: "F",
         1011: "R",
         1017: "S",
         1023: "L",
         1112: "F", 
         1213: "R",
         124: "L", 
         1310: "F", 
         1416: "F",
         1513: "S",
         1514: "R",
         1617: "R",
         169: "L",
         1623: "S",
         1718: "F",
         1819: "R",
         186: "L",
         1915: "F",
         2021: "R",
         2019: "S",
         2122: "F",
         2223: "R",
         229: "S",
         2211: "L",
         2320: "F",
         224: "F",
         924: "F",
         243: "F",
         425: "F",
         255: "F",
         1926: "F",
         2615: "F",
         628: "F",
         287: "F",
         2327: "F",
         2720: "F",}