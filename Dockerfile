# 1. Base image: a minimal Linux with Python 3.11 already installed
FROM python:3.11-slim


# 2. Working folder in the image (created if it does not exist)
WORKDIR /app


# 3. First copy ONLY the list of libraries, then install them.
# This layer is cached: as long as requirements.txt does not change,
# Docker does not redo the installation (saves several minutes).
COPY requirements.txt .
RUN pip install -r requirements.txt


# 4. Then copy the code and the model (they change more often)
COPY app.py .
COPY model/ model/


# 5. Default value: load the model from the model folder
ENV MODEL_URI=model


# 6. Document the port used by the API (information, publishes nothing)
EXPOSE 8000


# 7. Command run when the container starts.
# --host 0.0.0.0: listen on all the addresses of the container
# (with 127.0.0.1, the API would be unreachable from outside)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]