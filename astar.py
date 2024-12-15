class Container:
    """Representation of a container with name and weight."""
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        self.unload = False  # True if needs to be unloaded

class Slot:
    """Representation of a ship slot."""
    def __init__(self):
        self.container = None
        self.available = True

class Node():
    def __init__(self):
        self.ship_grid = None
        self.buffer = None
        self.timeCost = 0
        self.unloadHeuristic = self.unloadHeuristic()
        self.balanceHeuristic = self.balanceHeuristic()
        self.parent = None
        self.operator = ''
#grid = [[Slot() for _ in range(8)] for _ in range(12)]
# grid = [[0 for _ in range(12)] for _ in range(8)]
# grid[0][0] = Slot()
# grid[0][0].container = Container("Cat", 5)
# buffer = [[0 for _ in range(24)] for _ in range(4)]

# for row in grid[::-1]:
#     print(row)

 # enables checks for duplicate nodes to optimize search algorithm using == and 'in' keyword
def __eq__(self, object):
    return isinstance(object, Node) and (self.ship_grid == object.ship_grid) and (self.buffer == object.buffer)

# number of containers left to unload
def unloadHeuristic(ship_grid):
    heuristic = 0
    for row in ship_grid:
        for slot in row:
            if slot.container and slot.container.unload:
                heuristic += 1
            
    return heuristic

def balanceHeuristic(self, ship_grid):
    left_weight, right_weight, weight_balanced = calculate_balance(ship_grid)
    
    return abs(left_weight - right_weight)

# within buffer/ship, 1 min per slot
# between buffer and ship, 4 min
# between truck and buffer/ship, 2 min
# updating a container to be available?
def unload_expandNode(self):
    frontier = []
    for j in range(len(self.ship_grid[0])):
        # search column from top to bottom
        for i in reversed(range(len(self.ship_grid))):
            # if container exists and nothing is on top of it (top row edge index out of range edge case), apply operators to it
            if (self.ship_grid[i][j].container and i == 7) or (self.ship_grid[i][j].container and self.ship_grid[i+1][j].available):
                # prioritize unloading operator
                if self.ship_grid[i][j].container.unload:
                    currNode = Node()
                    currNode.ship_grid = self.ship_grid.copy()
                    currNode.ship_grid[i][j].container = None
                    currNode.ship_grid[i][j].available = True
                    if i != 0:
                        currNode.ship_grid[i-1][j].available = True
                    currNode.buffer = self.buffer.copy()
                    # time to get to top left corner + 1 to get out of ship, + 2 to truck
                    currNode.timeCost = self.timeCost + (7 - i) + j + 1 + 2
                    currNode.unloadHeuristic = currNode.ship_grid.unloadHeuristic()
                    currNode.balanceHeuristic = currNode.ship_grid.balanceHeuristic()
                    currNode.parent = self
                    currNode.operator = f'Unload container from [{i}, {j}] to loading area'
                    frontier.append(currNode)
                    break
                for col in range(len(self.ship_grid[0])):
                    # skip current column, not moving a container to its current position
                    if col != j:
                        row = find_slot_in_col(self, col)
                        # if column is not full
                        if row is not None:
                            currNode = Node()
                            currNode.ship_grid = self.ship_grid.copy()

                            # move container to new position
                            currNode.ship_grid[row][col] = Container()
                            currNode.ship_grid[row][col].container.name = self.ship_grid[i][j].container.name
                            currNode.ship_grid[row][col].container.weight = self.ship_grid[i][j].container.weight
                            currNode.ship_grid[row][col].container.unload = self.ship_grid[i][j].container.unload
                            if row != 0:
                                currNode.ship_grid[row-1][col].container.available = False
                            currNode.ship_grid[i][j].container = None
                            currNode.ship_grid[i][j].available = False
                            if i != 0:
                                currNode.ship_grid[i-1][j].available = True

                            # update costs and track operation
                            currNode.buffer = self.buffer.copy()
                            currNode.timeCost = self.timeCost + abs(i - row) + abs(j - col)
                            currNode.unloadHeuristic = currNode.ship_grid.unloadHeuristic()
                            currNode.balanceHeuristic = currNode.ship_grid.balanceHeuristic()
                            currNode.parent = self
                            currNode.operator = f'Move container from [{i}, {j}] to [{row}, {col}]'
                        frontier.append(currNode)

                # move to buffer operator
                currNode = Node()
                currNode.buffer = self.buffer.copy()
                buffer_x, buffer_y = self.buffer.find_available_slot()
                # move container to new position in buffer
                currNode.buffer[buffer_x][buffer_y] = Container()
                currNode.buffer[buffer_x][buffer_y].container.name = self.ship_grid[i][j].container.name
                currNode.buffer[buffer_x][buffer_y].container.weight = self.ship_grid[i][j].container.weight
                currNode.buffer[buffer_x][buffer_y].container.unload = self.ship_grid[i][j].container.unload
                if buffer_x != 0:
                    currNode.buffer[buffer_x-1][buffer_y].container.available = False
                currNode.ship_grid[i][j].container = None
                currNode.ship_grid[i][j].available = False
                if i != 0:
                    currNode.ship_grid[i-1][j].available = True

                # update costs and track operation
                currNode.ship_grid = self.ship_grid.copy()
                # time to get to top left corner + 1 to get out of ship, + 4 to buffer, + 1 to get in buffer + time to buffer slot
                currNode.timeCost = self.timeCost + abs(i - row) + abs(j - col) + 1 + 4 + 1 + (3 - buffer_y) + buffer_x
                currNode.unloadHeuristic = currNode.ship_grid.unloadHeuristic()
                currNode.balanceHeuristic = currNode.ship_grid.balanceHeuristic()
                currNode.parent = self
                currNode.operator = f'Move container from [{i}, {j}] to buffer [{buffer_x}, {buffer_y}]'
                frontier.append(currNode)
                # don't iterate rest of column when found container to apply operators
                break

    return frontier

def find_slot_in_col(self, col):
    for row in range(len(self.ship_grid)):
        if self.ship_grid[row][col].available:
            return row

    # column is full
    return None

def balance_expandNode(self):
    frontier = []
    
    return frontier

# return sequence of operations saved in a list
def unload_astar(ship_grid, buffer, unload_lst):
    operations = []

    initial_node = Node()
    initial_node.ship_grid = ship_grid.copy()
    initial_node.buffer = buffer.copy()
    currNode = None
    
    queue = [initial_node]
    expandedNodes = []
    while len(queue) > 0:
        currNode = queue[0]
        if currNode.unload_goal(currNode.ship_grid, currNode.buffer, unload_lst) == True:

            # add loading anything from buffer back to ship
            # TODO: buffer operations

            return operations
        newNodes = currNode.unload_expandNode()
        expandedNodes.append(queue.pop(0))
        # checks if expanded node has already been searched
        for node in newNodes:
            if node in expandedNodes:
                continue
            # some edge cases inserting into queue sorted by g(n) + h(n)
            if len(queue) == 0:
                queue.append(node)
                continue
            for i in range(-1, -len(queue), -1):
                if (node.timeCost + node.unloadHeuristic) > (queue[i].timeCost + queue[i].unloadHeuristic):
                    if i == -1:
                        queue.append(node)
                        break
                    queue.insert(i + 1, node)
                    break
            if i == -len(queue):
                queue.insert(0, node)
            
    # this really shouldn't ever return but just in case for debugging
    return ['No goal found']

# first unload, then load operations are straightforward
def unload_goal(ship_grid, buffer, unload_lst):
    for row in ship_grid:
        for slot in row:
            if slot.container and slot.container.name in unload_lst:
                return False
            
    return True

# return sequence of operations saved in a list
def balance_astar(ship_grid, buffer):
    operations = []
    
    # if frontier is empty, SIFT

    return operations

def balance_goal(ship_grid):
    left_weight, right_weight, weight_balanced = calculate_balance(ship_grid)

    return weight_balanced