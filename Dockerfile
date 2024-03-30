# Use the official Python image from Docker Hub
FROM python:3.8-slim

# Copy the Flask application code into the container
# Set the working directory inside the container
COPY . /app
WORKDIR /app

# Install dependencies using pip
RUN pip install -r requirements.txt

# Expose the port the Flask app runs on
EXPOSE 5000

# Define the command to run the Flask application when the container starts
CMD ["python3", "app.py"]
