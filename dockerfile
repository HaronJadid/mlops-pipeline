# 1. The Base Image: We start with a lightweight version of Python and Linux
FROM python:3.10-slim

# 2. The Working Directory: Create a folder inside the container to hold our app
WORKDIR /app

# 3. Copy Dependencies: Copy only the requirements file first 
# (This is a Docker optimization trick so it doesn't reinstall packages if only your code changes)
COPY requirements.txt .

# 4. Install Dependencies: Run pip install inside the container
# --no-cache-dir keeps the container size small by not saving the downloaded installation files
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Application: Copy your code and your trained model into the container
COPY app.py .
COPY model.pt .

# 6. Expose the Port: Tell Docker that the container will listen on port 8000
EXPOSE 8000

# 7. The Wake-Up Command: The CMD instruction we discussed yesterday.
# This runs Uvicorn on host 0.0.0.0 (meaning it accepts connections from outside the container)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]