# Dockerfile

# Use the official Python base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Copy files to the container
COPY . /app

# Update the package lists
RUN apt-get update \
    && apt-get install -y python3 python3-pip libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the app will run
EXPOSE 7860

# Set the entrypoint command to run the app.py file
CMD ["python3", "app.py"]
