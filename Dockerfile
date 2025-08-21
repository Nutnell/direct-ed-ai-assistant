# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# The port the app will run on
EXPOSE 8000

# The command to run the application
# We use --host 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]