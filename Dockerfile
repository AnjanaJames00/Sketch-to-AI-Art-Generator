# Use a lightweight Python base image
FROM python:3.9-slim
# Set the working directory inside the
container
WORKDIR /app
# Copy the Python script into the container
COPY hello.py .
# Command to run the Python script
CMD ["python", "hello.py"]