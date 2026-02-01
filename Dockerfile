FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway에서 부여하는 PORT 환경변수를 사용하도록 변경
CMD sh -c "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"