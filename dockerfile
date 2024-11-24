# Use an official lightweight Python image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on (default 8888)
EXPOSE 8888

# Set environment variables (optional defaults)
ENV PORT=8888
ENV ADDRESS=0.0.0.0
ENV MODEL_PATH=full_pn.pt

# Command to run the application
CMD ["python", "pneumonia_detection.py"]
