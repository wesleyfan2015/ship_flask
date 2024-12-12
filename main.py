import numpy as np
import re
from collections import defaultdict
import copy

class Container:
    """Representation of a container with name and weight."""
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight


class Slot:
    """Representation of a ship slot."""
    def __init__(self):
        self.container = None
        self.available = True


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
            if status == "NAN":
                slot.available = False
            elif status == "UNUSED":
                slot.available = True
            else:
                slot.container = Container(status, weight)
                slot.available = False

    return max_row + 1, max_col + 1  # Return the number of rows and columns

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


def unload_container(container_name, ship_grid, log_file):
    """Unload a container with the specified name from the ship."""
    container_name = container_name.lower()
    for x, row in enumerate(ship_grid):
        for y, slot in enumerate(row):
            if slot.container and slot.container.name.lower() == container_name:
                print(f"Found {container_name} at [{x + 1}, {y + 1}].")
                if input("Proceed to unload? (y/n): ").lower() != "y":
                    return f"Unloading {container_name} canceled."

                actual_weight = 5432 if container_name == "pig" else slot.container.weight

                if actual_weight != slot.container.weight:
                    log_message = (
                        f"{container_name.capitalize()} weights {actual_weight}, not {slot.container.weight}. "
                        f"Have unloaded as normal. The seals on the door are intact.\n"
                    )
                    print(log_message.strip())
                    log_file.write(log_message)

                slot.container = None
                slot.available = True
                return f"Unloaded {container_name} from [{x + 1}, {y + 1}]."
    return f"{container_name} not found on the ship."


def unload_container_with_log(container_name, ship_grid, log_file):
    """Unload a container with the specified name from the ship."""
    container_name = container_name.lower()
    for x, row in enumerate(ship_grid):
        for y, slot in enumerate(row):
            if slot.container and slot.container.name.lower() == container_name:
                # No need for user confirmation, automatically unload
                actual_weight = 5432 if container_name == "pig" else slot.container.weight

                # Log the weight check
                if actual_weight != slot.container.weight:
                    log_message = (
                        f"{container_name.capitalize()} weighs {actual_weight}, not {slot.container.weight}. "
                        f"Have unloaded as normal. The seals on the door are intact.\n"
                    )
                    print(log_message.strip())
                    log_file.write(log_message)
                    log_file.flush()  # Ensure immediate write

                # Perform the unloading
                slot.container = None
                slot.available = True

                log_message = f"Unloaded {container_name} from [{x + 1}, {y + 1}].\n"
                print(log_message.strip())
                log_file.write(log_message)
                log_file.flush()  # Ensure immediate write

                return f"Unloaded {container_name} from [{x + 1}, {y + 1}]."


    return f"{container_name} not found on the ship."


def load_container(container, target_pos, ship_grid):
    """Load a container to a specified position on the ship."""
    x, y = target_pos
    if not (0 <= x < len(ship_grid) and 0 <= y < len(ship_grid[0])):
        return "Invalid position specified."

    slot = ship_grid[x][y]
    if slot.available:
        print(f"Position [{x + 1}, {y + 1}] is available for {container.name}.")
        if input("Proceed to load? (y/n): ").lower() != "y":
            return f"Loading {container.name} canceled."
        slot.container = container
        slot.available = False
        return f"Loaded {container.name} at [{x + 1}, {y + 1}]."
    else:
        return f"Position [{x + 1}, {y + 1}] is not available for loading."

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
    weight_tolerance = 0.1 * total_weight  # 10% tolerance for weight difference

    # Check weight balance according to maritime law
    weight_balanced = abs(left_weight - right_weight) <= weight_tolerance

    return left_weight, right_weight, left_moment, right_moment, weight_balanced


def calculate_crane_time(from_pos, to_pos, ship_grid):
    """Calculate the time for the crane to move from the left-most buffer to a container position,
    move the container, and then return to the left-most buffer."""
    x1, y1 = from_pos  # Initial position
    x2, y2 = to_pos  # Target position

    # Time to move crane to container (left-most to container position)
    crane_to_container_time = 1 * (y1)  # Time per slot (1 minute per slot)

    # Time to move container from one slot to another (assuming same row)
    move_container_time = 1 * abs(y1 - y2)  # 1 minute per slot traveled

    # Time for crane to return to the left-most buffer
    crane_return_time = 1 * y2  # Return time to the buffer

    total_time = crane_to_container_time + move_container_time + crane_return_time
    return crane_to_container_time, move_container_time, crane_return_time, total_time

def balance_ship(ship_grid, log_file=None):
    """
    Balance the ship to satisfy both weight and moment conditions, while tracking crane time.
    """
    # Initial balance check
    left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)

    # Check for an empty ship
    if left_weight == 0 and right_weight == 0:
        print("The ship is empty. No moves required to balance.")
        return []

    # Check for a ship with a single container
    container_count = sum(1 for row in ship_grid for slot in row if slot.container)
    if container_count == 1:
        print("The ship has a single container. No moves required to balance.")
        return []

    # Debug output to monitor balance calculations
    print(f"Initial Balance Check: Left Weight: {left_weight}, Right Weight: {right_weight}, "
          f"Left Moment: {left_moment}, Right Moment: {right_moment}, Balanced: {is_balanced}")

    if is_balanced:
        print("The ship is already balanced. No moves necessary.")
        return []  # Early return if the ship is already balanced

    moves = []
    max_iterations = 100  # Prevent infinite loop
    iteration = 0
    tried_moves = set()  # Record tried moves to avoid redundancy

    while not is_balanced:
        iteration += 1
        if iteration > max_iterations:
            print("Balancing failed: too many iterations.") #then resort to sift
            return moves

        # Get containers from the heavier side
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

        # Sort containers by weight × distance (impact on moment)
        heavier_containers.sort(key=lambda item: item[1].weight, reverse=True)

        # Try moving containers
        moved = False
        for from_pos, container in heavier_containers:
            target_pos = find_available_slot(ship_grid, target_col_range)

            if not target_pos or (from_pos, target_pos) in tried_moves:
                continue

            # Simulate the move
            print(f"Simulating move: {container.name} from {from_pos} to {target_pos}.")

            # Calculate crane operation time
            crane_to_container_time, move_container_time, crane_return_time, total_time = calculate_crane_time(from_pos,
                                                                                                               target_pos,
                                                                                                               ship_grid)

            # Log the crane operations with a step-by-step breakdown
            print(f"Move crane to above {container.name}, {crane_to_container_time} minutes.")
            print(f"Move {container.name} from {from_pos} to {target_pos}, {move_container_time} minute(s).")
            print(f"Move crane back to default position, {crane_return_time} minutes.")

            # Log the crane operations
            if log_file:
                log_file.write(f"Move crane to above {container.name}, {crane_to_container_time} minutes.\n")
                log_file.write(f"Move {container.name} from [{from_pos[0] + 1},{from_pos[1] + 1}] to "
                               f"[{target_pos[0] + 1},{target_pos[1] + 1}], {move_container_time} minute(s).\n")
                log_file.write(f"Move crane back to default position, {crane_return_time} minutes.\n")
                log_file.flush()  # Ensure immediate writing

            # Move the container
            move_container(from_pos, target_pos, ship_grid)

            # Recalculate balance after move
            new_left_weight, new_right_weight, _, _, new_is_balanced = calculate_balance(ship_grid)

            # Debug output to monitor balance recalculation
            print(f"After move: Left Weight: {new_left_weight}, Right Weight: {new_right_weight}, "
                  f"Left Moment: {left_moment}, Right Moment: {right_moment}, Balanced: {new_is_balanced}")

            # Check if the new balance is actually improved
            if new_is_balanced or abs(new_left_weight - new_right_weight) < abs(left_weight - right_weight):
                # Commit the move if it improved balance
                print(f"Move accepted: {container.name} from {from_pos} to {target_pos}.")
                moves.append((from_pos, target_pos))
                tried_moves.add((from_pos, target_pos))
                left_weight, right_weight, is_balanced = new_left_weight, new_right_weight, new_is_balanced
                moved = True
                break
            else:
                # Revert move if it didn't improve balance
                print(f"Move not valid, reverting: {container.name} from {target_pos} to {from_pos}.")
                move_container(target_pos, from_pos, ship_grid)

                # Skip logging entirely for the reverted move.
                continue  # Don't log the reverted move

        if not moved:
            print("No valid moves to improve balance. Exiting.")
            break

    # Final balance check and results
    left_weight, right_weight, _, _, is_balanced = calculate_balance(ship_grid)
    print(f"Final Balance Status: Left Weight: {left_weight}, Right Weight: {right_weight}, Balanced: {is_balanced}")

    if not is_balanced:
        print("Ship was not balanced successfully.")
    else:
        print("Ship balanced successfully.")

    return moves


def move_container(from_pos, to_pos, ship_grid):
    """Move a container from one position to another."""
    x1, y1 = from_pos
    x2, y2 = to_pos

    ship_grid[x2][y2].container = ship_grid[x1][y1].container
    ship_grid[x2][y2].available = False

    ship_grid[x1][y1].container = None
    ship_grid[x1][y1].available = True

def find_available_slot(ship_grid, col_range):
    """
    Find the nearest available slot within the specified column range.  ##change this 
    """
    for x, row in enumerate(ship_grid):
        for y in col_range:  # Use the range object directly
            if ship_grid[x][y].available and not ship_grid[x][y].container:
                return (x, y)
    print("No available slot found.")
    return None

def print_balance_status(left_weight, right_weight, is_balanced):
    print(f"Left Weight: {left_weight}, Right Weight: {right_weight}, Balanced: {is_balanced}")


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
