# Use official Python slim image
FROM python:3.9-slim

# Prevent .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Set default port environment variable
ENV PORT 8080

# Run the Flask app
CMD ["python", "app.py"]
