# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the local code to the container
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose any ports (optional, depends on your app's need)
EXPOSE 5000

# Define the command to run your app
CMD ["python", "app.py"]