FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set default command to run the application
CMD ["python3", "inference.py"]