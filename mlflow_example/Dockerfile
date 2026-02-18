FROM python:3.12-slim

# Install git, which might be needed for some pip packages
RUN apt-get update && apt-get install -y git

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy the rest of the application code
COPY . .
