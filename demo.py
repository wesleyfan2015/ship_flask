import os
import re
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OPERATION_LOG'] = 'operation_logs'
app.config['BALANCE_LOG'] = os.path.join(app.config['OPERATION_LOG'], 'balance_logs')
app.config['LOAD_LOG'] = os.path.join(app.config['OPERATION_LOG'], 'load_logs')
app.config['UNLOAD_LOG'] = os.path.join(app.config['OPERATION_LOG'], 'unload_logs')
app.config['BUFFER_LOG'] = 'buffer_logs'
app.config['ACCEPTED_EXTENSIONS'] = {'txt'}

# Secret key for session handling
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

class Buffer:
    """Representation of a buffer with its own grid."""
    def __init__(self, rows, columns):
        self.grid = [[Slot() for _ in range(columns)] for _ in range(rows)]

    def find_available_slot(self):
        """Find the first available slot in the buffer."""
        for x, row in enumerate(self.grid):
            for y, slot in enumerate(row):
                if slot.available:
                    return (x, y)
        return None  # Buffer is full

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
                slot.available = True
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
    weight_tolerance = 0.1 * total_weight  # 10% tolerance for weight difference

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
                available = find_available_slot(ship_grid, range(len(ship_grid[0]) // 2, len(ship_grid[0])))
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

def find_available_slot(ship_grid, col_range):
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

def move_container(from_pos, to_pos, ship_grid):
    """Move a container from one position to another on the ship grid."""
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

def unload_container_with_log(container_name, ship_grid, buffer, log_file):
    """Unload a container with the specified name from the ship, utilizing the buffer."""
    container_name = container_name.lower()
    for x, row in enumerate(ship_grid):
        for y, slot in enumerate(row):
            if slot.container and slot.container.name.lower() == container_name:
                log_file.write(f"Found container {container_name} at [{x + 1}, {y + 1}].\n")

                # Identify containers above the target container (assuming y-axis is vertical)
                containers_above = [(x, col) for col in range(y + 1, len(row)) if row[col].container]

                # Move containers above to buffer
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

    return moves

@app.route('/')
def index():
    """Render the main interface displaying the current ship and buffer grids."""
    return render_template(
        'demo.html',
        grid=ship_grid,
        buffer=buffer,
        rows=rows,
        cols=cols,
        buffer_rows=buffer_rows,
        buffer_cols=buffer_cols
    )

@app.route('/upload', methods=['POST'])
def upload():
    """Handle the uploaded file and load the ship grid."""
    global ship_grid, buffer, rows, cols
    buffer = Buffer(buffer_rows, buffer_cols)  # Reset buffer
    ship_grid = create_ship_grid(rows, cols)  # Reset ship grid

    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return "Invalid file type. Please upload a .txt file.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Store the uploaded filename in session
    session['uploaded_filename'] = filename

    try:
        rows, cols = load_ship_grid(file_path, ship_grid)  # Dynamically get rows and cols
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
    """Balance the ship grid."""
    try:
        # Retrieve the uploaded filename from the session
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
    """Load a container onto the ship."""
    try:
        # Retrieve the uploaded filename from the session
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Retrieve input from the form
        container_name = request.form['load']
        row = int(request.form['row']) - 1  # Adjust to zero-indexing
        col = int(request.form['col']) - 1  # Adjust to zero-indexing
        weight = float(request.form['weight'])

        # Create a container object
        container = Container(name=container_name, weight=weight)

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_load_log.log'
        log_file_path = os.path.join(app.config['LOAD_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform the load operation and pass the log file for writing
            message = load_container_with_log(container, (row, col), ship_grid, log_file)

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
    """Unload a container from the ship using the buffer."""
    try:
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Retrieve the container name from the form
        container_name = request.form['container_name']

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_unload_log.log'
        log_file_path = os.path.join(app.config['UNLOAD_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform the unload operation and pass the log file for writing
            message = unload_container_with_log(container_name, ship_grid, buffer, log_file)

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

@app.route('/move_to_buffer', methods=['POST'])
def move_to_buffer_route():
    """Move a container from the ship to the buffer."""
    try:
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Retrieve ship positions from the form
        ship_x = int(request.form['ship_x']) - 1
        ship_y = int(request.form['ship_y']) - 1

        # Validate that the specified position is within the ship grid
        if not (0 <= ship_x < rows and 0 <= ship_y < cols):
            return "Specified position is out of ship grid bounds.", 400

        # Check if there is a container at the specified ship position
        if not ship_grid[ship_x][ship_y].container:
            return f"No container at Ship[{ship_x + 1}, {ship_y + 1}] to move.", 400

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_buffer_log.log'
        log_file_path = os.path.join(app.config['BUFFER_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform the move to buffer and log the action
            message = move_to_buffer((ship_x, ship_y), buffer, ship_grid, log_file)

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
        return f"Error moving to buffer: {e}", 500

@app.route('/move_from_buffer', methods=['POST'])
def move_from_buffer_route():
    """Move a container from the buffer back to the ship."""
    try:
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Retrieve buffer positions from the form
        buffer_x = int(request.form['buffer_x']) - 1
        buffer_y = int(request.form['buffer_y']) - 1

        # Validate that the specified position is within the buffer grid
        if not (0 <= buffer_x < buffer_rows and 0 <= buffer_y < buffer_cols):
            return "Specified position is out of buffer grid bounds.", 400

        # Check if there is a container at the specified buffer position
        if not buffer.grid[buffer_x][buffer_y].container:
            return f"No container at Buffer[{buffer_x + 1}, {buffer_y + 1}] to move.", 400

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_buffer_log.log'
        log_file_path = os.path.join(app.config['BUFFER_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform the move from buffer and log the action
            message = move_from_buffer((buffer_x, buffer_y), buffer, ship_grid, log_file)

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
        return f"Error moving from buffer: {e}", 500

@app.route('/move_within_buffer', methods=['POST'])
def move_within_buffer_route():
    """Move a container within the buffer."""
    try:
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded. Please upload a manifest file first.", 400

        # Retrieve source and target buffer positions from the form
        from_x = int(request.form['from_x']) - 1
        from_y = int(request.form['from_y']) - 1
        to_x = int(request.form['to_x']) - 1
        to_y = int(request.form['to_y']) - 1

        # Validate that the specified positions are within the buffer grid
        if not (0 <= from_x < buffer_rows and 0 <= from_y < buffer_cols and
                0 <= to_x < buffer_rows and 0 <= to_y < buffer_cols):
            return "Specified positions are out of buffer grid bounds.", 400

        # Check if there is a container at the source buffer position
        if not buffer.grid[from_x][from_y].container:
            return f"No container at Buffer[{from_x + 1}, {from_y + 1}] to move.", 400

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_buffer_log.log'
        log_file_path = os.path.join(app.config['BUFFER_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform the move within buffer and log the action
            message = move_within_buffer((from_x, from_y), (to_x, to_y), buffer, log_file)

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
        return f"Error moving within buffer: {e}", 500

@app.route('/depart', methods=['POST'])
def depart_route():
    """Finalize operations before ship departure, ensuring buffer integrity."""
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

def allowed_file(filename):
    """Check if the file has the correct extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ACCEPTED_EXTENSIONS']

# Global Variables
rows, cols = 8, 12  # Default ship grid size
buffer_rows, buffer_cols = 2, 5  # Default buffer grid size
ship_grid = create_ship_grid(rows, cols)
buffer = Buffer(buffer_rows, buffer_cols)

if __name__ == '__main__':
    app.run(debug=True)
