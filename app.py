from flask import Flask, request, render_template, jsonify,session
from werkzeug.utils import secure_filename
import os

from main import *
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OPERATION_LOG'] = 'balance_logs'
app.config['LOAD_LOG'] = 'load_logs'
app.config['UNLOAD_LOG'] = 'unload_logs'
app.config['ACCEPT_EXTENSION'] = 'txt'

# Secret key for session handling
app.secret_key = 'adcdefghijklmnopqrstuvwxyz'

global rows, cols, ship_grid, uploaded_filename

# Initialize the ship grid
rows, cols = 8, 12
ship_grid = create_ship_grid(rows, cols)
uploaded_filename = None  # To store the filename of the uploaded file


def allowed_file(filename):
    """Check if the file has the correct extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ACCEPT_EXTENSION']

@app.route('/')
def index():
    # init the grid table
    ship_grid = create_ship_grid(rows, cols)
    return render_template('index.html', grid=ship_grid, rows=rows, cols=cols)

@app.route('/upload', methods=['POST'])
def upload():
    ship_grid.clear()
    """Handle the uploaded file and load the ship grid."""
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return "Invalid file type", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Store the uploaded filename in session
    session['uploaded_filename'] = filename  # Store the file name in session

    try:
        rows, cols = load_ship_grid(file_path, ship_grid)  # Get rows and cols dynamically
        return render_template('index.html', grid=ship_grid, rows=rows, cols=cols)
    except Exception as e:
        return f"Error loading ship grid: {e}", 500


@app.route('/balance', methods=['POST'])
def balance():
    """Balance the ship grid."""
    try:
        # Retrieve the uploaded filename from the session
        uploaded_filename = session.get('uploaded_filename')  # Retrieve the file name from session
        if not uploaded_filename:
            return "No file uploaded", 400

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_balance_log.log'

        log_file_path = os.path.join(app.config['OPERATION_LOG'], log_file_name)  # Save log in the operation log folder

        with open(log_file_path, 'a') as log_file:  # Open the log file in append mode
            # Perform the balance operation and pass the log file for writing
            balance_ship(ship_grid, log_file=log_file)

        # Return the updated grid with rows and columns
        return render_template('index.html', grid=ship_grid, rows=rows, cols=cols)
    except Exception as e:
        return f"Error balancing ship grid: {e}", 500

@app.route('/load', methods=['POST'])
def load():
    """Load the ship grid."""
    try:
        # Retrieve the uploaded filename from the session
        uploaded_filename = session.get('uploaded_filename')  # Retrieve the file name from session
        if not uploaded_filename:
            return "No file uploaded", 400

        # Retrieve input from the form
        container_name = request.form['load']
        row = int(request.form['row']) - 1  # Adjust to zero-indexing
        col = int(request.form['col']) - 1  # Adjust to zero-indexing
        weight = float(request.form['weight'])

        # Create a container object (make sure it exists or is created as per your model)
        container = Container(name=container_name, weight=weight)

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_load_log.log'

        log_file_path = os.path.join(app.config['LOAD_LOG'], log_file_name)  # Save log in the operation log folder

        with open(log_file_path, 'a') as log_file:
            # Perform the load operation and pass the log file for writing
            message = load_container_with_log(container, (row, col), ship_grid, log_file)

        # Return the updated grid with rows and columns
        return render_template('index.html', grid=ship_grid, rows=len(ship_grid), cols=len(ship_grid[0]), message=message)
    except Exception as e:
        return f"Error loading ship grid: {e}", 500

@app.route('/unload', methods=['POST'])
def unload():
    """Unload the container from the ship."""
    try:
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            return "No file uploaded", 400

        # Retrieve the container name from the form
        container_name = request.form['container_name']

        log_file_name = uploaded_filename.rsplit('.', 1)[0] + '_unload_log.log'
        log_file_path = os.path.join(app.config['UNLOAD_LOG'], log_file_name)

        with open(log_file_path, 'a') as log_file:
            # Perform the unload operation and pass the log file for writing
            message = unload_container_with_log(container_name, ship_grid, log_file)

        # Return the updated grid with rows and columns
        return render_template('index.html', grid=ship_grid, rows=rows, cols=cols, message=message)
    except Exception as e:
        return f"Error unloading container: {e}", 500


if __name__ == '__main__':
    app.run()
