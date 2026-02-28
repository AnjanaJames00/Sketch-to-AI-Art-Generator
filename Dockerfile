# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    torch \
    diffusers \
    transformers \
    opencv-python \
    gradio \
    pillow \
    numpy \
    accelerate

# Gradio default port
EXPOSE 7860

# Run the app
CMD ["python", "AIDraw.py"]