from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from datetime import datetime

from main import *
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OPERATION_LOG'] = 'balance_logs'
app.config['LOAD_LOG'] = 'load_logs'
app.config['UNLOAD_LOG'] = 'unload_logs'
app.config['ACCEPT_EXTENSION'] = 'txt'

# session handling
app.secret_key = 'adcdefghijklmnopqrstuvwxyz'

global rows, cols, ship_grid, uploaded_filename

# ship grid display
rows, cols = 8, 12
ship_grid = create_ship_grid(rows, cols)

# manifest
uploaded_filename = None 


def allowed_file(filename):
    """Check if the file has the correct extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ACCEPT_EXTENSION']

# web page submission file
@app.route('/')
def login():
    """Render the login page as the default page."""
    return render_template('login.html')


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Handles the Load/Unload button and renders demo.html."""
    return render_template(
        'demo.html',
        grid=ship_grid,
        buffer=buffer,
        rows=rows,
        cols=cols,
        buffer_rows=buffer_rows,
        buffer_cols=buffer_cols
    )

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Handles login page"""
    return render_template(
        'login.html',
) 

# load/unload and balance options (and logout button)
@app.route('/ship_options')
def ship_options():
    return render_template('ship_options.html')

@app.route('/load_unload')
def load_unload():
    ship_grid = create_ship_grid(rows, cols)
    return render_template('load_unload.html', grid=ship_grid, rows=rows, cols=cols)

@app.route('/logout')
def logout():
    return redirect(url_for('loginPage'))

@app.route('/ship_problem')
def index():
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

    session['uploaded_filename'] = filename

    try:
        #  load the ship grid from the file path
        rows, cols = load_ship_grid(file_path, ship_grid)  
        return render_template(
            'upload.html',
            message="File uploaded and ship grid loaded successfully!",
            uploaded_file=filename,  
            grid=ship_grid,
            buffer=buffer,
            rows=rows,
            cols=cols
        )
    except Exception as e:
        return render_template(
            'upload.html',
            error=f"Error loading ship grid: {e}"
        )

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
