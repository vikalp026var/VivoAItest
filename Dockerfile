# Use the official Python image from Docker Hub
FROM python:3.8

# Set environment variables
ENV PORT=8000

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the app runs
EXPOSE $PORT

# Command to run the application using Gunicorn
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:${PORT}", "app:app"]
