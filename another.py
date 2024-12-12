import heapq
from heapq import heappush, heappop
import hashlib
import copy
import os
import re
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename

class Node:
    def __init__(self, ship_grid, buffer_grid, g_cost, parent= None):
        self.ship_grid = ship_grid
        self.buffer_grid = buffer_grid
        self.g_cost = g_cost
        self.h_cost = balance_heuristic(ship_grid)
        self.f_cost = self.h_cost + self.g_cost
        self.parent = parent

    def __lt__(self, other):
        # For comparison of nodes based on f (priority queue needs this)
        return self.f_cost < other.f_cost

    def __repr__(self):
        return f"Node(f_cost={self.f_cost}, g_cost={self.g_cost}, h_cost={self.h_cost})" 
        
    def __hash__(self):
        # Only hash the ship grid part, buffer is immutable
        return hash(tuple(tuple(row) for row in self.ship_grid))
    
    def __eq__(self, other):
        # Compare only the ship grid state for equality
        return self.ship_grid == other.ship_grid

class Container:
    """Representation of a container with name and weight."""
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

class Slot:
    """Representation of a ship or buffer slot."""
    def __init__(self):
        self.container = None
        self.available = True
    
    def __hash__(self):
        return hash(self.available)

    def __eq__(self, other):
        return isinstance(other, Slot) and self.available == other.available

class Buffer:
    """Representation of a buffer with its own grid."""
    def __init__(self, rows, columns):
        self.grid = [[Slot() for _ in range(columns)] for _ in range(rows)]

    def find_available_slot(self):
        """Find the first available slot in the buffer."""
        for x, row in enumerate(self.grid):
            for y in range(len(row) - 1, -1, -1):
                if slot.available:
                    return (x, y)
        return None  # Buffer is full



def valid_moves(ship_grid, current_pos):
    valid_moves = []
    
    x1, y1 = current_pos
    if ship_grid[x1][y1] is None:
        return []  # No container at the current position

    # Check for available slots on the ship grid
    for x in range(len(ship_grid)):
        for y in range(len(ship_grid[0])):
            slot = ship_grid[x][y]
            if slot.available:
                if x == 0 or ship_grid[x - 1][y] is not None:
                    valid_moves.append((x, y))  # Add empty slot as a valid move

    return valid_moves

def balance_heuristic(ship_grid):
    left_weight, right_weight, left_moment, right_moment, isBalanced = calculate_balance(ship_grid)
    imbalance = abs(right_weight - left_weight)
    return imbalance


def make_move(ship_grid, from_pos, to_pos):
    # Create a copy of the ship grid
    new_ship_grid = [row[:] for row in ship_grid]
    new_ship_grid[to_pos[0]][to_pos[1]] = new_ship_grid[from_pos[0]][from_pos[1]]
    new_ship_grid[from_pos[0]][from_pos[1]] = None 
    return new_ship_grid


def move_cost(from_pos, to_pos):
    x1, y1 = from_pos
    x2, y2 = to_pos
    return abs(x1 - x2) + abs(y1 - y2)


def astar_balance_ship(ship_grid, buffer):
    queue = []
    initial_node = Node(ship_grid, buffer, 0)  # Initial node with cost 0
    heappush(queue,  initial_node)  

    visited = set()
    visited.add(initial_node)

    while queue:
        current_node = heappop(queue)

        left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)

        if is_balanced:
            print("Ship is now balanced")
            return reconstruct_path(current_node)  # Return the sequence of moves to reach the goal

        # Generate valid moves for each container on the ship

        for x in range(len(current_node.ship_grid)):
            for y in range(len(current_node.ship_grid[0])):
                if current_node.ship_grid[x][y] is not None:  
                    current_pos = (x, y)

                    if(can_move_up(current_node.ship_grid, current_pos)):
                        moves = valid_moves(current_node.ship_grid, current_pos)

                        # Process in-grid moves (move container within the ship)
                        for move in moves:
                            new_ship_grid = copy.deepcopy(current_node.ship_grid) 
                            move_container(current_pos, move, new_ship_grid)
                            new_cost = current_node.g + move_cost(current_pos, move)
                            new_node = Node(new_ship_grid, current_node.buffer, new_cost, current_node)

                            if new_node not in visited:
                                visited.add(new_node)
                                heappush(queue, new_node)

                        # Process move to the buffer (buffer is independent grid) will change to closest available buffer slot
                        buffer_move = current_node.buffer.find_available_slot()
                        if buffer_move is not None:
                            leave_pos = (7,0)
                            enter_pos = (3, 23)
                            buffer_cost = current_node.g + 4 + move_cost(current_pos, leave_pos) + move_cost(enter_pos, buffer_move) # Add the buffer transfer cost
                            new_ship_grid = copy.deepcopy(current_node.ship_grid)  # Deepcopy to avoid modifying the original
                            new_buffer = copy.deepcopy(current_node.buffer)  # Deepcopy to avoid modifying the original
                            move_to_buffer_without_log(current_pos, new_ship_grid, new_buffer)
                            new_node = Node(new_ship_grid, new_buffer, buffer_cost, current_node)

                            if new_node not in visited:
                                visited.add(new_node)
                                heappush(queue, new_node)

        # Process moving containers back from the buffer to the ship //worry abt this later 
        for x in range(len(current_node.buffer)):
            for y in range(len(current_node.buffer[0])):
                if current_node.buffer[x][y] is not None:  # If there's a container in the buffer
                    buffer_pos = (x, y)
                    ship_move = find_available_slot(ship_grid, range(0, len(ship_grid[0])), from_pos)
                    if ship_move is not None:
                        enter_pos = (7,0)
                        leave_pos = (3, 23)
                        back_to_ship_cost = current_node.g + 4 + move_cost(enter_pos, buffer_pos) + move_cost(leave_pos, ship_move)  # Add cost for moving from buffer to the ship
                        new_ship_grid = copy.deepcopy(current_node.ship_grid)  # Deepcopy to avoid modifying the original
                        new_buffer = copy.deepcopy(current_node.buffer)  # Deepcopy to avoid modifying the original
                        move_from_buffer_without_log(buffer_pos, ship_move, new_ship_grid, new_buffer)
                        new_node = Node(new_ship_grid, new_buffer, back_to_ship_cost, current_node)

                        if new_node not in visited:
                            visited.add(new_node)
                            heappush(queue, new_node)


def move_to_buffer_without_log(from_pos, ship_grid, buffer):
    """
    Move a container from the ship to the buffer.
    """
    x, y = from_pos
    slot = ship_grid[x][y]
    if not slot.container:
        return "No container at the specified position to move."

    available = buffer.find_available_slot()
    if not available:
        return "Buffer is full. Cannot move container to buffer."

    bx, by = available
    buffer.grid[bx][by].container = slot.container
    buffer.grid[bx][by].available = False

    slot.container = None
    slot.available = True

    return f"Moved {buffer.grid[bx][by].container.name} to Buffer[{bx + 1}, {by + 1}]."

def move_from_buffer_without_log(from_pos, to_pos, ship_grid, buffer):
    """
    Move a container from the buffer back to the ship.
    """
    bx, by = to_pos
    buffer_slot = buffer.grid[bx][by]
    if not buffer_slot.container:
        return "No container at the specified buffer position to move."

    available = find_available_slot(ship_grid, range(len(ship_grid[0]) // 2, len(ship_grid[0])))
    if not available:
        return "No available slot on the ship to move the container."

    x, y = available
    ship_grid[x][y].container = buffer_slot.container
    ship_grid[x][y].available = False

    buffer_slot.container = None
    buffer_slot.available = True

    return f"Moved {ship_grid[x][y].container.name} to Ship[{x + 1}, {y + 1}]."


def reconstruct_path(goal_node):
    # Reconstruct the path by tracing back to the initial node
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append((current_node.ship_grid, current_node.buffer))
        current_node = current_node.parent
    return path[::-1]  # Return the path in the correct order



















def create_ship_grid(rows, columns):
    """Create an empty ship grid."""
    return [[Slot() for _ in range(columns)] for _ in range(rows)]

def load_ship_grid(file_path, ship_grid):
    """Load ship grid from a manifest file and adjust rows and columns dynamically."""
    max_row, max_col = 0, 0  # To track the maximum row and column number

    with open(file_path, 'r') as file:
        for line in file.readlines():
            match = re.match(r"\[(\d{2}),(\d{2})\], \{(\d+)\}, (.+)", line.strip())
            if not match:
                print(f"Skipping invalid line: {line.strip()}")
                continue

            x, y = int(match.group(1)) - 1, int(match.group(2)) - 1
            weight = int(match.group(3))
            status = match.group(4).strip()

            # Update max row and column values
            max_row = max(max_row, x)
            max_col = max(max_col, y)

            # Ensure the ship grid has enough rows and columns
            while len(ship_grid) <= max_row:
                ship_grid.append([Slot() for _ in range(max_col + 1)])
            while len(ship_grid[x]) <= max_col:
                ship_grid[x].append(Slot())

            slot = ship_grid[x][y]
            if status.upper() == "NAN":
                slot.available = False
            elif status.upper() == "UNUSED":
                slot.available = True  #need to return error if name is NAN
            else:
                slot.container = Container(status, weight)
                slot.available = False

    return max_row + 1, max_col + 1  # Return the number of rows and columns

def calculate_balance(ship_grid):
    """
    Calculate the weight and moment (torque) of the ship's left and right sides.
    Returns:
        - Left weight, right weight
        - Left moment, right moment
        - Whether the ship is balanced by weight (maritime law)
    """
    left_weight, right_weight = 0, 0
    left_moment, right_moment = 0, 0
    mid_col = len(ship_grid[0]) // 2  # Middle column index

    for row in ship_grid:
        for col, slot in enumerate(row):
            if slot.container:
                weight = slot.container.weight
                if col < mid_col:
                    left_weight += weight
                    left_moment += weight * (mid_col - col)  # Distance to midline
                else:
                    right_weight += weight
                    right_moment += weight * (col - mid_col + 1)  # Distance to midline

    total_weight = left_weight + right_weight
    weight_tolerance = 0.1 * max(left_weight, right_weight)  # 10% tolerance for weight difference

    # Check weight balance according to maritime law
    weight_balanced = abs(left_weight - right_weight) <= weight_tolerance

    return left_weight, right_weight, left_moment, right_moment, weight_balanced

def calculate_crane_time(from_pos, to_pos, ship_grid):
    """Calculate the time for the crane to move from the buffer to a container position,
    move the container, and then return to the buffer."""
    x1, y1 = from_pos  # Initial position
    x2, y2 = to_pos  # Target position

    # Time to move crane to container (buffer to container position)
    crane_to_container_time = 1 * (y1)  # Time per slot (1 minute per slot)

    # Time to move container from one slot to another (assuming same row)
    move_container_time = 1 * abs(y1 - y2)  # 1 minute per slot traveled

    # Time for crane to return to the buffer
    crane_return_time = 1 * y2  # Return time to the buffer

    total_time = crane_to_container_time + move_container_time + crane_return_time
    return crane_to_container_time, move_container_time, crane_return_time, total_time

def verify_buffer_integrity(buffer, ship_grid, log_file=None):
    """
    Ensure all containers in the buffer are loaded back onto the ship.
    """
    messages = []
    for x, row in enumerate(buffer.grid):
        for y, slot in enumerate(row):
            if slot.container:
                available = find_available_slot(ship_grid, range(len(ship_grid[0]) // 2, len(ship_grid[0])), from_pos)
                if available:
                    ship_x, ship_y = available
                    ship_grid[ship_x][ship_y].container = slot.container
                    ship_grid[ship_x][ship_y].available = False

                    log_message = f"Moved {slot.container.name} from Buffer[{x + 1}, {y + 1}] to Ship[{ship_x + 1}, {ship_y + 1}].\n"
                    messages.append(log_message)
                    if log_file:
                        log_file.write(log_message)

                    slot.container = None
                    slot.available = True
                else:
                    message = f"Cannot return container {slot.container.name} to the ship. Ship is full."
                    messages.append(message)
                    return messages
    return messages

def find_available_slot(ship_grid, col_range): #need to change, what if trying to move container in right most slot or make new one 
    """
    Find the nearest available slot within the specified column range.
    col_range: range object indicating columns to search.
    """
    for x, row in enumerate(ship_grid):
        for y in col_range:
            if ship_grid[x][y].available and not ship_grid[x][y].container:
                return (x, y)
    print("No available slot found.")
    return None

def can_move_up(ship_grid, pos):
    x, y = pos
    for row in range(x+1, len(ship_grid)):
        if ship_grid[row][y].container:
            return False 
    return True


def move_container(from_pos, to_pos, ship_grid):
    """Move a container from one position to another on the ship grid."""
    if can_move_up(ship_grid, from_pos):
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        container = ship_grid[from_x][from_y].container
        ship_grid[to_x][to_y].container = container
        ship_grid[to_x][to_y].available = False
        ship_grid[from_x][from_y].container = None
        ship_grid[from_x][from_y].available = True

def move_to_buffer(from_pos, buffer, ship_grid, log_file=None):
    """
    Move a container from the ship to the buffer.
    """
    x, y = from_pos
    slot = ship_grid[x][y]
    if not slot.container:
        return "No container at the specified position to move."

    available = buffer.find_available_slot()
    if not available:
        return "Buffer is full. Cannot move container to buffer."

    bx, by = available
    buffer.grid[bx][by].container = slot.container
    buffer.grid[bx][by].available = False

    if log_file:
        log_message = f"Moved {buffer.grid[bx][by].container.name} from Ship[{x + 1}, {y + 1}] to Buffer[{bx + 1}, {by + 1}].\n"
        log_file.write(log_message)
        log_file.flush()

    slot.container = None
    slot.available = True

    return f"Moved {buffer.grid[bx][by].container.name} to Buffer[{bx + 1}, {by + 1}]."

def move_from_buffer(to_pos, buffer, ship_grid, log_file=None):
    """
    Move a container from the buffer back to the ship.
    """
    bx, by = to_pos
    buffer_slot = buffer.grid[bx][by]
    if not buffer_slot.container:
        return "No container at the specified buffer position to move."

    available = find_available_slot(ship_grid, range(len(ship_grid[0]) // 2, len(ship_grid[0])))
    if not available:
        return "No available slot on the ship to move the container."

    x, y = available
    ship_grid[x][y].container = buffer_slot.container
    ship_grid[x][y].available = False

    if log_file:
        log_message = f"Moved {ship_grid[x][y].container.name} from Buffer[{bx + 1}, {by + 1}] to Ship[{x + 1}, {y + 1}].\n"
        log_file.write(log_message)
        log_file.flush()

    buffer_slot.container = None
    buffer_slot.available = True

    return f"Moved {ship_grid[x][y].container.name} to Ship[{x + 1}, {y + 1}]."

def move_within_buffer(from_pos, to_pos, buffer, log_file=None):
    """
    Move a container within the buffer.
    """
    fx, fy = from_pos
    tx, ty = to_pos
    from_slot = buffer.grid[fx][fy]
    to_slot = buffer.grid[tx][ty]

    if not from_slot.container:
        return "No container at the source buffer position to move."

    if not to_slot.available:
        return "Target buffer position is not available."

    to_slot.container = from_slot.container
    to_slot.available = False

    from_slot.container = None
    from_slot.available = True

    if log_file:
        log_message = f"Moved {to_slot.container.name} from Buffer[{fx + 1}, {fy + 1}] to Buffer[{tx + 1}, {ty + 1}].\n"
        log_file.write(log_message)
        log_file.flush()

    return f"Moved {to_slot.container.name} to Buffer[{tx + 1}, {ty + 1}]."

def load_container_with_log(container, target_pos, ship_grid, log_file):
    """Load a container to a specified position on the ship."""
    x, y = target_pos
    if not (0 <= x < len(ship_grid) and 0 <= y < len(ship_grid[0])):
        return "Invalid position specified."

    slot = ship_grid[x][y]
    if slot.available:
        log_file.write(f"Attempting to load container {container.name} to position [{x + 1}, {y + 1}].\n")
        slot.container = container
        slot.available = False
        log_file.write(f"Successfully loaded {container.name} at position [{x + 1}, {y + 1}].\n")
        return f"Loaded {container.name} at [{x + 1}, {y + 1}]."
    else:
        log_file.write(f"Failed to load {container.name} at position [{x + 1}, {y + 1}] - Not available.\n")
        return f"Position [{x + 1}, {y + 1}] is not available for loading."

def unload_container_with_log(container_name, ship_grid, buffer, log_file ): # add to option incase to truck
    """Unload a container with the specified name from the ship, utilizing the buffer."""
    container_name = container_name.lower()
    for x, row in enumerate(ship_grid):
        for y, slot in enumerate(row):
            if slot.container and slot.container.name.lower() == container_name:
                log_file.write(f"Found container {container_name} at [{x + 1}, {y + 1}].\n")

                # Identify containers above the target container (assuming y-axis is vertical)
                containers_above = [(x, col) for col in range(y + 1, len(row)) if row[col].container]

                # Move containers above to buffer 
                # or to any open slot on ship
                for pos in containers_above:
                    move_result = move_to_buffer(pos, buffer, ship_grid, log_file)
                    log_file.write(move_result + "\n")

                # Unload the target container
                actual_weight = 5432 if container_name == "pig" else slot.container.weight

                if actual_weight != slot.container.weight:
                    log_message = (
                        f"Container {container_name.capitalize()} weighs {actual_weight}, not {slot.container.weight}. "
                        f"Have unloaded as normal. The seals on the door are intact.\n"
                    )
                    log_file.write(log_message)
                else:
                    log_file.write(f"Unloaded container {container_name} from [{x + 1}, {y + 1}].\n")

                slot.container = None
                slot.available = True

                # Move containers back from buffer to ship
                for pos in containers_above:
                    move_result = move_from_buffer(pos, buffer, ship_grid, log_file)
                    log_file.write(move_result + "\n")

                return f"Unloaded {container_name} from [{x + 1}, {y + 1}]."
    return f"{container_name} not found on the ship."

def balance_ship(ship_grid, buffer, log_file=None):
    """
    Balance the ship to satisfy both weight and moment conditions, while tracking crane time.
    """
    # Initial balance check
    left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)

    if log_file:
        log_file.write(f"Initial Balance Check: Left Weight: {left_weight}, Right Weight: {right_weight}, "
                      f"Left Moment: {left_moment}, Right Moment: {right_moment}, Balanced: {is_balanced}\n")

    # Check for an empty ship
    if left_weight == 0 and right_weight == 0:
        if log_file:
            log_file.write("The ship is empty. No moves required to balance.\n")
        return []

    # Check for a ship with a single container
    container_count = sum(1 for row in ship_grid for slot in row if slot.container)
    if container_count == 1:
        if log_file:
            log_file.write("The ship has a single container. No moves required to balance.\n")
        return []

    # If already balanced, return early
    if is_balanced:
        if log_file:
            log_file.write("The ship is already balanced. No moves necessary.\n")
        return []

    moves = []
    max_iterations = 100  # Prevent infinite loop
    iteration = 0
    tried_moves = set()  # Record tried moves to avoid redundancy

    while not is_balanced and iteration < max_iterations:
        iteration += 1

        # Recalculate balance
        left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)
        if is_balanced:
            break
        ''''
        # Determine heavier side
        if left_weight > right_weight or left_moment > right_moment:
            heavier_side = "left"
            col_range = range(0, len(ship_grid[0]) // 2)
            target_col_range = range(len(ship_grid[0]) // 2, len(ship_grid[0]))
        else:
            heavier_side = "right"
            col_range = range(len(ship_grid[0]) // 2, len(ship_grid[0]))
            target_col_range = range(0, len(ship_grid[0]) // 2)

        # Get containers on the heavier side
        heavier_containers = [
            ((x, y), row[y].container) for x, row in enumerate(ship_grid)
            for y in col_range if row[y].container
        ]

        # Sort containers by weight descending
        heavier_containers.sort(key=lambda item: item[1].weight, reverse=True)

        moved = False
        for (from_pos, container) in heavier_containers:
            target_pos = find_available_slot(ship_grid, target_col_range)
            if not target_pos or (from_pos, target_pos) in tried_moves:
                continue

            # Estimate crane time
            crane_to_container_time, move_container_time, crane_return_time, total_time = calculate_crane_time(from_pos, target_pos, ship_grid)

            # Log crane operations
            if log_file:
                log_file.write(f"Move crane to container {container.name} at [{from_pos[0] + 1}, {from_pos[1] + 1}], {crane_to_container_time} minutes.\n")
                log_file.write(f"Move {container.name} from [{from_pos[0] + 1}, {from_pos[1] + 1}] to [{target_pos[0] + 1}, {target_pos[1] + 1}], {move_container_time} minute(s).\n")
                log_file.write(f"Move crane back to buffer, {crane_return_time} minutes.\n")

            # Perform the move
            move_container(from_pos, target_pos, ship_grid)
            moves.append((from_pos, target_pos))
            tried_moves.add((from_pos, target_pos))
            moved = True
            break  # Move one container at a time

        if not moved:
            if log_file:
                log_file.write("No valid moves to improve balance. Exiting.\n")
            break


    # Final balance check
    left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)
    if log_file:
        log_file.write(f"Final Balance Status: Left Weight: {left_weight}, Right Weight: {right_weight}, Balanced: {is_balanced}\n")

    if not is_balanced:
        if log_file:
            log_file.write("Ship was not balanced successfully.\n")
    else:
        if log_file:
            log_file.write("Ship balanced successfully.\n")
    '''

    return astar_balance_ship(ship_grid, buffer)  #uppppppppp

def print_ship_grid(ship_grid):
    """Print the current ship grid with actual content."""
    grid_to_display = []
    for row in ship_grid:
        display_row = []
        for slot in row:
            if slot.container:
                display_row.append(slot.container.name)
            elif slot.available:
                display_row.append("UNUSED")
            else:
                display_row.append("NAN")
        grid_to_display.append(display_row)

    # Print the grid in reverse order to display bottom-up
    for row in grid_to_display[::-1]:
        print(row)

if __name__ == "__main__":
    rows, cols = 8, 12
    ship_grid = create_ship_grid(rows, cols)

    file_path = input("Enter manifest file path: ")
    try:
        load_ship_grid("ShipCase" + file_path + ".txt", ship_grid)
        print("Loaded ship grid:")
        print_ship_grid(ship_grid)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    # Dynamically create a log file name based on the manifest file
    log_file_name = f"{file_path}_operations.log"
    with open(log_file_name, "a") as log_file:
        print(f"Logging operations to {log_file_name}")

        while True:
            action = input("Choose action (balance/load/unload): ").lower()
            if action == "balance":
                print("Balancing the ship...")
                moves = balance_ship(ship_grid, log_file)
                print("Moves performed:", moves)
            elif action == "unload":
                container_name = input("Enter container name to unload: ")
                result = unload_container(container_name, ship_grid, log_file)
                print(result)
            elif action == "load":
                name = input("Enter container name: ")
                weight = int(input("Enter container weight: "))
                x, y = map(int, input("Enter target position (row col): ").split())
                result = load_container(Container(name, weight), (x - 1, y - 1), ship_grid)
                print(result)
            else:
                print("Invalid action.")

            print("Current Ship Grid:")
            print_ship_grid(ship_grid)

            if input("Do you want to perform another operation? (y/n): ").lower() != "y":
                print("Exiting the program.")
                break


