# Dockerfile - minimal container for a Streamlit app
FROM python:3.10-slim

# set workdir
WORKDIR /app

# install system deps (if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# copy files
COPY . /app

# upgrade pip and install python deps
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# streamlit server config environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_RUN_ON_SAVE=true
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# start streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
