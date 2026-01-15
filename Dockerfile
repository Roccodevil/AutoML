# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Install Java (Required for H2O)
# We update sources, install the default Java Runtime (JRE), and clean up cache to keep it small.
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Set the working directory in the container
WORKDIR /app

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install Python dependencies
# We use --no-cache-dir to keep the image lightweight
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# 6. Copy the rest of the application code
COPY . .

# 7. Expose the port Render uses (10000 is default for their Docker runtime)
EXPOSE 10000

# 8. Define the command to run the app
# We point Gunicorn to the 'run' file we will create in Step 2
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]