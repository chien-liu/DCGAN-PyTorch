FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install the project (and its dependencies)
RUN pip install --upgrade pip \
    && pip install .

# Entrypoint allows user to pass args to the CLI script
ENTRYPOINT ["gan_face_generate"]
