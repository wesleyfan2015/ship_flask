import os
import re
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OPERATION_LOG'] = 'operation_logs'
app.config['BALANCE_LOG'] = os.path.join(app.config['OPERATION_LOG'], 'balance_logs')
app.config['LOAD_LOG'] = os.path.join(app.config['OPERATION_LOG'], 'load_logs')
app.config['UNLOAD_LOG'] = os.path.join(app.config['OPERATION_LOG'], 'unload_logs')
app.config['BUFFER_LOG'] = 'buffer_logs'
app.config['ACCEPTED_EXTENSIONS'] = {'txt'}

# Secret key for session management
app.secret_key = 'adcdefghijklmnopqrstuvwxyz'

# Ensure necessary directories exist
for directory in [
    app.config['UPLOAD_FOLDER'],
    app.config['BALANCE_LOG'],
    app.config['LOAD_LOG'],
    app.config['UNLOAD_LOG'],
    app.config['BUFFER_LOG']
]:
    os.makedirs(directory, exist_ok=True)

# container
class Container:
    """Represents a container with a name, weight, and unload flag."""

    def __init__(self, name, weight, unload=False):
        self.name = name
        self.weight = weight
        self.unload = unload  # True if needs to be unloaded

# slots in ship grid
class Slot:
    """Represents a slot in the ship or buffer grid."""

    def __init__(self):
        self.container = None
        self.available = True

# buffer that contains the containers
class Buffer:
    """Represents a buffer area with its own grid."""

    def __init__(self, rows, columns):
        self.grid = [[Slot() for _ in range(columns)] for _ in range(rows)]

    def find_available_slot(self):
        """Finds the first available slot in the buffer starting from the bottom."""
        for x in reversed(range(len(self.grid))):  # Start from the bottom
            for y in range(len(self.grid[x])):
                if self.grid[x][y].available:
                    return (x, y)
        return None  # Buffer is full

# create the ship grid
def create_ship_grid(rows, columns):
    """Creates an empty ship grid with specified number of rows and columns."""
    return [[Slot() for _ in range(columns)] for _ in range(rows)]

# load the ship grid
def load_ship_grid(file_path, ship_grid):
    """Loads the ship grid from a manifest file, dynamically adjusting rows and columns."""
    max_row, max_col = 0, 0  # Track maximum row and column indices

    with open(file_path, 'r') as file:
        for line in file.readlines():
            print(f"Processing line: {line.strip()}")
            # Update regex to match three fields
            match = re.match(r"\[(\d{2}),(\d{2})\], \{(\d+)\}, (.+)", line.strip())
            if not match:
                print(f"Skipping invalid line: {line.strip()}")
                continue

            x, y = int(match.group(1)) - 1, int(match.group(2)) - 1
            weight = int(match.group(3))
            status = match.group(4).strip()
            unload_flag = False  # Default no unload

            # Update maximum row and column values
            max_row = max(max_row, x)
            max_col = max(max_col, y)

            # Ensure ship grid has enough rows and columns
            while len(ship_grid) <= max_row:
                ship_grid.append([Slot() for _ in range(max_col + 1)])
            while len(ship_grid[x]) <= max_col:
                ship_grid[x].append(Slot())

            slot = ship_grid[x][y]
            if status.upper() == "NAN":
                slot.available = False
            elif status.upper() == "UNUSED":
                slot.available = True
            else:
                slot.container = Container(status, weight, unload=unload_flag)
                slot.available = False

    return max_row + 1, max_col + 1  # Return number of rows and columns

# calculate the weight balance from middle to left and right
def calculate_balance(ship_grid):
    """
    Calculates the balance of the ship based on weight
    Returns:
        - Left weight, Right weight
        - Whether the ship is balanced (according to maritime regulations)
    """
    left_weight, right_weight = 0, 0
    mid_col = len(ship_grid[0]) // 2  # Center line column index

    for row in ship_grid:
        for col, slot in enumerate(row):
            if slot.container:
                weight = slot.container.weight
                if col < mid_col:
                    left_weight += weight
                else:
                    right_weight += weight

    weight_tolerance = 0.1 * max(left_weight, right_weight)  # 10% weight difference tolerance

    # Check weight balance according to maritime regulations
    weight_balanced = abs(left_weight - right_weight) <= weight_tolerance
    
    return left_weight, right_weight, weight_balanced

# calculate how many time moving the crane between buffer and ship grid
def calculate_crane_time(from_pos, to_pos, ship_grid):
    """Calculates the crane time required to move a container from buffer to ship."""
    x1, y1 = from_pos  # Starting position
    x2, y2 = to_pos  # Target position

    # Time for crane to move to container position (1 minute per slot)
    crane_to_container_time = 1 * y1

    # Time to move the container (assuming same row, 1 minute per slot)
    move_container_time = 1 * abs(y1 - y2)

    # Time for crane to return to buffer
    crane_return_time = 1 * y2

    total_time = crane_to_container_time + move_container_time + crane_return_time
    return crane_to_container_time, move_container_time, crane_return_time, total_time

# verify the buffer integrity
def verify_buffer_integrity(buffer, ship_grid, log_file=None):
    """
    Ensures that all containers in the buffer that do not need to be unloaded are loaded back onto the ship.
    """
    messages = []
    for x, row in enumerate(buffer.grid):
        for y, slot in enumerate(row):
            if slot.container:
                container = slot.container
                if not container.unload:
                    # Find available slot in the ship (right half)
                    available = find_available_slot(ship_grid, range(len(ship_grid[0]) // 2, len(ship_grid[0])))
                    if available:
                        ship_x, ship_y = available
                        ship_grid[ship_x][ship_y].container = container
                        ship_grid[ship_x][ship_y].available = False

                        log_message = f"Moved {container.name} from Buffer[{x + 1}, {y + 1}] to Ship[{ship_x + 1}, {ship_y + 1}].\n"
                        messages.append(log_message)
                        if log_file:
                            log_file.write(log_message)

                        # Clear the buffer slot
                        slot.container = None
                        slot.available = True
                    else:
                        message = f"Cannot return container {container.name} to the ship. Ship is full."
                        messages.append(message)
                        if log_file:
                            log_file.write(message + "\n")
                        continue
    return messages

# find all possible slot in the ship grid.
def find_available_slot(ship_grid, col_range):
    """
    Finds the nearest available slot within the specified column range.
    col_range: A range object indicating which columns to search.
    """
    for x, row in enumerate(ship_grid):
        for y in col_range:
            if ship_grid[x][y].available and not ship_grid[x][y].container:
                return (x, y)
    print("No available slot found.")
    return None

# find if there is any container in the buffer area that can be move back into the ship grid
def find_first_movable_container(buffer, ship_grid, log_file=None):
    """
    Finds the first container in the buffer that is not marked for unloading and can be moved to the ship.
    Returns: ((x, y), container) or None
    """
    for x in reversed(range(len(buffer.grid))):  # Start from the bottom of the buffer
        for y in range(len(buffer.grid[x])):
            slot = buffer.grid[x][y]
            if slot.container and not slot.container.unload:
                if log_file:
                    log_file.write(
                        f"Found container '{slot.container.name}' at Buffer[{x + 1}, {y + 1}] marked for moving.\n")
                # Find available slot in the ship (right half)
                available_slot = find_available_slot(ship_grid, range(len(ship_grid[0]) // 2, len(ship_grid[0])))
                if available_slot:
                    if log_file:
                        log_file.write(
                            f"Available ship slot found at Ship[{available_slot[0] + 1}, {available_slot[1] + 1}].\n")
                    return ((x, y), slot.container)
                else:
                    if log_file:
                        log_file.write("No available ship slots found for this container.\n")
    if log_file:
        log_file.write("No movable containers found in buffer.\n")
    return None

# function to move the container from one position to another
def move_container(from_pos, to_pos, ship_grid):
    """Moves a container within the ship grid from one position to another."""
    from_x, from_y = from_pos
    to_x, to_y = to_pos
    container = ship_grid[from_x][from_y].container
    ship_grid[to_x][to_y].container = container
    ship_grid[to_x][to_y].available = False
    ship_grid[from_x][from_y].container = None
    ship_grid[from_x][from_y].available = True

# load container into ship gird, if the ship gird is 50% full, move the container to the buffer
def load_container_with_log(container, target_pos, ship_grid, buffer, log_file):
    """
    Loads a container into the specified position on the ship, or moves it to the buffer based on capacity.
    """
    x, y = target_pos
    if not (0 <= x < len(ship_grid) and 0 <= y < len(ship_grid[0])):
        return "Invalid position specified."

    available_slots = calculate_available_slots(ship_grid)
    total_slots = len(ship_grid) * len(ship_grid[0])
    load_threshold = 0.5  # Threshold set to 50%, can be adjusted as needed

    log_file.write(f"Attempting to load container {container.name} with weight {container.weight}.\n")
    log_file.write(f"Current available slots: {available_slots}/{total_slots}.\n")

    # Check if the ship is nearing full capacity
    if available_slots / total_slots < load_threshold:
        log_file.write(f"Ship is near full capacity. Moving container {container.name} to buffer.\n")
        # Attempt to move container to buffer
        available = buffer.find_available_slot()
        if available:
            bx, by = available
            buffer.grid[bx][by].container = container
            buffer.grid[bx][by].available = False
            # Preserve unload flag based on container's original setting
            buffer.grid[bx][by].container.unload = container.unload
            log_file.write(f"Successfully moved {container.name} to Buffer[{bx + 1}, {by + 1}].\n")
            return f"Container {container.name} moved to Buffer[{bx + 1}, {by + 1}] because the ship is near full capacity."
        else:
            log_file.write(f"Buffer is full. Cannot move {container.name} to buffer.\n")
            return "Buffer is full. Cannot load the container."

    # If the ship has space, continue loading the container
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

# unload the container into the void
def unload_container_with_log(container_name, ship_grid, log_file):
    """
    Unloads the specified container from the ship grid and completely removes it from the system.
    """
    for x, row in enumerate(ship_grid):
        for y, slot in enumerate(row):
            if slot.container and slot.container.name == container_name:
                # Directly remove the container without moving to buffer
                slot.container = None
                slot.available = True

                log_message = f"Unloaded container '{container_name}' from Ship[{x + 1}, {y + 1}] and removed it from the system.\n"
                log_file.write(log_message)

                return f"Unloaded container '{container_name}' from Ship[{x + 1}, {y + 1}] and removed it from the system."

    log_message = f"Container '{container_name}' not found in ship grid.\n"
    log_file.write(log_message)
    return f"Container '{container_name}' not found in ship grid."

# count the number of available slots in the ship gird
def calculate_available_slots(ship_grid):
    """Calculates the number of available slots in the ship grid."""
    available = 0
    for row in ship_grid:
        for slot in row:
            if slot.available:
                available += 1
    return available

# balance the ship gird
def balance_ship(ship_grid, buffer, log_file=None):
    """
    Balances the ship to meet weight and moment conditions while tracking crane time.
    Also attempts to load containers from buffer back to ship grid if they do not need to be unloaded.
    """
    # Initial balance check
    left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)

    if log_file:
        log_file.write(f"Initial Balance Check: Left Weight: {left_weight}, Right Weight: {right_weight}, "
                       f"Left Moment: {left_moment}, Right Moment: {right_moment}, Balanced: {is_balanced}\n")

    # Check if the ship is empty
    if left_weight == 0 and right_weight == 0:
        if log_file:
            log_file.write("The ship is empty. No moves required to balance.\n")
        return []

    # Check if there is only one container on the ship
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
    max_iterations = 100  # Prevent infinite loops
    iteration = 0
    tried_moves = set()  # Track tried moves to avoid repetition

    while not is_balanced and iteration < max_iterations:
        iteration += 1
        if log_file:
            log_file.write(f"Balance Iteration {iteration}\n")

        # Recalculate balance
        left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)
        if log_file:
            log_file.write(f"Recalculated Balance: Left Weight: {left_weight}, Right Weight: {right_weight}, "
                           f"Left Moment: {left_moment}, Right Moment: {right_moment}, Balanced: {is_balanced}\n")
        if is_balanced:
            break

        # Determine the heavier side
        if left_weight > right_weight or left_moment > right_moment:
            heavier_side = "left"
            col_range = range(0, len(ship_grid[0]) // 2)
            target_col_range = range(len(ship_grid[0]) // 2, len(ship_grid[0]))
        else:
            heavier_side = "right"
            col_range = range(len(ship_grid[0]) // 2, len(ship_grid[0]))
            target_col_range = range(0, len(ship_grid[0]) // 2)

        if log_file:
            log_file.write(f"Heavier side determined: {heavier_side}\n")

        # Get containers on the heavier side
        heavier_containers = [
            ((x, y), row[y].container) for x, row in enumerate(ship_grid)
            for y in col_range if row[y].container
        ]

        if not heavier_containers:
            if log_file:
                log_file.write("No containers found on the heavier side to move.\n")
            break

        # Sort containers by weight in descending order
        heavier_containers.sort(key=lambda item: item[1].weight, reverse=True)

        if log_file:
            container_names = [container.name for _, container in heavier_containers]
            log_file.write(f"Heavier containers (sorted by weight): {container_names}\n")

        moved = False
        for (from_pos, container) in heavier_containers:
            # Use find_available_slot instead of find_available_ship_slot
            available_slot = find_available_slot(ship_grid, target_col_range)
            if not available_slot or (from_pos, available_slot) in tried_moves:
                continue

            # Check unload flag
            if container.unload:
                if log_file:
                    log_file.write(f"Container '{container.name}' marked for unloading. Skipping.\n")
                continue

            # Estimate crane time
            crane_to_container_time, move_container_time, crane_return_time, total_time = calculate_crane_time(
                from_pos, available_slot, ship_grid)

            # Log crane operations
            if log_file:
                log_file.write(
                    f"Move crane to container {container.name} at [{from_pos[0] + 1}, {from_pos[1] + 1}], {crane_to_container_time} minutes.\n")
                log_file.write(
                    f"Move {container.name} from [{from_pos[0] + 1}, {from_pos[1] + 1}] to [{available_slot[0] + 1}, {available_slot[1] + 1}], {move_container_time} minute(s).\n")
                log_file.write(f"Move crane back to buffer, {crane_return_time} minutes.\n")

            # Perform the move
            move_container(from_pos, available_slot, ship_grid)
            moves.append((from_pos, available_slot))
            tried_moves.add((from_pos, available_slot))
            if log_file:
                log_file.write(f"Moved container '{container.name}' from {from_pos} to {available_slot}.\n")
            moved = True
            break  # Move one container at a time

        if not moved:
            if log_file:
                log_file.write("No valid moves performed in this iteration. Exiting.\n")
            break

    # Attempt to load containers from buffer back to ship grid
    buffer_loaded = False
    for x in range(len(buffer.grid)):
        for y in range(len(buffer.grid[x])):
            buffer_slot = buffer.grid[x][y]
            if buffer_slot.container:
                container = buffer_slot.container
                # Only load containers not marked for unloading
                if not container.unload:
                    available = find_available_slot(ship_grid, target_col_range)
                    if available:
                        ship_x, ship_y = available
                        ship_grid[ship_x][ship_y].container = container
                        ship_grid[ship_x][ship_y].available = False
                        buffer_slot.container = None
                        buffer_slot.available = True
                        log_message = f"Moved {container.name} from Buffer[{x + 1}, {y + 1}] to Ship[{ship_x + 1}, {ship_y + 1}].\n"
                        if log_file:
                            log_file.write(log_message)
                        buffer_loaded = True
    if buffer_loaded and log_file:
        log_file.write("Loaded containers from buffer to ship where possible.\n")

    # Final balance check
    left_weight, right_weight, left_moment, right_moment, is_balanced = calculate_balance(ship_grid)
    if log_file:
        log_file.write(
            f"Final Balance Status: Left Weight: {left_weight}, Right Weight: {right_weight}, Balanced: {is_balanced}\n")

    if not is_balanced:
        if log_file:
            log_file.write("Ship was not balanced successfully.\n")
    else:
        if log_file:
            log_file.write("Ship balanced successfully.\n")

    return moves


@app.route('/')
def index():
    """Renders the main interface displaying the current ship and buffer grids."""
    return render_template(
        'demo.html',
        # blow are the interaction required for the demo.html
        # basically just two sets of girds, one for ship and one for buffer
        grid=ship_grid,
        buffer=buffer,
        rows=rows,
        cols=cols,
        buffer_rows=buffer_rows,
        buffer_cols=buffer_cols
    )


@app.route('/upload', methods=['POST'])
def upload():
    """Handles file upload and loads the ship grid."""
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return "Invalid file type. Please upload a .txt file.", 400

    filename = secure_filename(file.filename)
    global file_path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Store the uploaded filename in the session
    # After stored in session, it can be brought to other page.
    session['uploaded_filename'] = filename

    return redirect(url_for('ship_options.html'))

@app.route('/load_unload', methods=['POST'])
def load_unload():
    global ship_grid, buffer, rows, cols
    buffer = Buffer(buffer_rows, buffer_cols)  # Reset buffer
    ship_grid = create_ship_grid(rows, cols)  # Reset ship grid

    try:
        rows, cols = load_ship_grid(file_path, ship_grid)  # Dynamically get rows and columns
        return render_template(
            'demo.html',
            grid=ship_grid,
            buffer=buffer,
            rows=rows,
            cols=cols,
            buffer_rows=buffer_rows,
            buffer_cols=buffer_cols
        )
    except Exception as e:
        return f"Error loading ship grid: {e}", 500

@app.route('/balance', methods=['POST'])
def balance_route():
    """Balances the ship grid."""
    try:
        # Get the uploaded filename from the session
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_balance_log.log'
        log_file_path = os.path.join(app.config['BALANCE_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            moves = balance_ship(ship_grid, buffer, log_file)

        return render_template(
            'demo.html',
            grid=ship_grid,
            buffer=buffer,
            rows=rows,
            cols=cols,
            buffer_rows=buffer_rows,
            buffer_cols=buffer_cols,
            message=f"Balancing completed. Moves performed: {moves}"
        )
    except Exception as e:
        return f"Error balancing ship grid: {e}", 500


@app.route('/load', methods=['POST'])
def load_route():
    """Loads a container into the ship."""
    try:
        # Get the uploaded filename from the session
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Get input from the form
        container_name = request.form['load']
        row = int(request.form['row']) - 1  # Adjust to zero index
        col = int(request.form['col']) - 1  # Adjust to zero index
        weight = float(request.form['weight'])
        unload_flag = request.form.get('unload_flag', 'false').lower() == 'true'  # Get unload flag from form, default False

        # Create a container object
        container = Container(name=container_name, weight=weight, unload=unload_flag)

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_load_log.log'
        log_file_path = os.path.join(app.config['LOAD_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform load operation and log
            message = load_container_with_log(container, (row, col), ship_grid, buffer, log_file)

        return render_template(
            'demo.html',
            grid=ship_grid,
            buffer=buffer,
            rows=rows,
            cols=cols,
            buffer_rows=buffer_rows,
            buffer_cols=buffer_cols,
            message=message
        )
    except Exception as e:
        return f"Error loading ship grid: {e}", 500


@app.route('/unload', methods=['POST'])
def unload_route():
    """Unloads a container from the ship, completely removing it."""
    try:
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Get container name from form
        container_name = request.form['container_name']

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_unload_log.log'
        log_file_path = os.path.join(app.config['UNLOAD_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform unload operation and log
            message = unload_container_with_log(container_name, ship_grid, log_file)

        return render_template(
            'demo.html',
            grid=ship_grid,
            buffer=buffer,
            rows=rows,
            cols=cols,
            buffer_rows=buffer_rows,
            buffer_cols=buffer_cols,
            message=message
        )
    except Exception as e:
        return f"Error unloading container: {e}", 500


@app.route('/depart', methods=['POST'])
def depart_route():
    """Ensures buffer integrity before ship departure."""
    try:
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_departure_log.log'
        log_file_path = os.path.join(app.config['BALANCE_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            integrity_messages = verify_buffer_integrity(buffer, ship_grid, log_file)
            for msg in integrity_messages:
                log_file.write(msg + "\n")

        if any("Cannot return" in msg for msg in integrity_messages):
            return render_template(
                'demo.html',
                grid=ship_grid,
                buffer=buffer,
                rows=rows,
                cols=cols,
                buffer_rows=buffer_rows,
                buffer_cols=buffer_cols,
                message="Departure failed: " + integrity_messages[0]
            )

        balance_message = "Ship is departing with all buffer containers properly loaded."
        return render_template(
            'demo.html',
            grid=ship_grid,
            buffer=buffer,
            rows=rows,
            cols=cols,
            buffer_rows=buffer_rows,
            buffer_cols=buffer_cols,
            message=balance_message
        )
    except Exception as e:
        return f"Error during departure: {e}", 500


@app.route('/next', methods=['POST'])
def next_route():
    """Handles moving a container from the buffer to the ship grid."""
    try:
        # Get the uploaded filename from the session
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Set the log file path
        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_next_log.log'
        log_file_path = os.path.join(app.config['LOAD_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Find the first movable container in the buffer
            container_info = find_first_movable_container(buffer, ship_grid, log_file)
            if not container_info:
                message = "No movable containers in buffer."
                log_file.write(message + "\n")
                return render_template(
                    'demo.html',
                    grid=ship_grid,
                    buffer=buffer,
                    rows=rows,
                    cols=cols,
                    buffer_rows=buffer_rows,
                    buffer_cols=buffer_cols,
                    message=message
                )

            (bx, by), container = container_info

            # Find available slot in the ship (right half)
            ship_slot = find_available_slot(ship_grid, range(len(ship_grid[0]) // 2, len(ship_grid[0])))
            if not ship_slot:
                message = "No available slots in ship grid to move the container."
                log_file.write(message + "\n")
                return render_template(
                    'demo.html',
                    grid=ship_grid,
                    buffer=buffer,
                    rows=rows,
                    cols=cols,
                    buffer_rows=buffer_rows,
                    buffer_cols=buffer_cols,
                    message=message
                )

            sx, sy = ship_slot

            # Perform the move operation
            ship_grid[sx][sy].container = container
            ship_grid[sx][sy].available = False

            buffer.grid[bx][by].container = None
            buffer.grid[bx][by].available = True

            # Log the move
            log_message = f"Moved container '{container.name}' from Buffer[{bx + 1}, {by + 1}] to Ship[{sx + 1}, {sy + 1}].\n"
            log_file.write(log_message)

            message = f"Moved container '{container.name}' to Ship[{sx + 1}, {sy + 1}]."
            return render_template(
                'demo.html',
                grid=ship_grid,
                buffer=buffer,
                rows=rows,
                cols=cols,
                buffer_rows=buffer_rows,
                buffer_cols=buffer_cols,
                message=message
            )
    except Exception as e:
        return f"Error during next operation: {e}", 500


def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ACCEPTED_EXTENSIONS']


# Global variables
rows, cols = 8,12  # Ship grid size (rows x columns)
buffer_rows, buffer_cols = 4, 24  # Buffer grid size (rows x columns)
ship_grid = create_ship_grid(rows, cols)
buffer = Buffer(buffer_rows, buffer_cols)

if __name__ == '__main__':
    app.run(debug=True)
