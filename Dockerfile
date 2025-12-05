FROM python:3.11-slim

WORKDIR /app

# ---- System dependencies required by WeasyPrint ----
RUN apt-get update && apt-get install -y \
    build-essential \
    libcairo2 \
    libcairo2-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    libxml2 \
    libxslt1.1 \
    fonts-dejavu-core \
    shared-mime-info \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Copy requirements & install Python packages ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application ----
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
