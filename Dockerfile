# Use the official Python 3.12 slim image as the base (lightweight)
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including app.py, templates, and .pkl files) into the container
COPY . .

# Expose port 5000 (Flask default port)
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]