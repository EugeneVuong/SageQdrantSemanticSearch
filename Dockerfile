FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
