# Use an official Python runtime as the base image
FROM python:3.10.16-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    libportaudio2 \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Create a dummy ALSA sound configuration
RUN echo "pcm.!default { type plug slave.pcm null }" > /etc/asound.conf

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]