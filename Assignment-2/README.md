# 🚀 FastAPI Dockerized Backend API

This project is a simple backend API built using FastAPI and containerized with Docker as part of an assignment.

---

## 📌 Features

- Basic REST API using FastAPI
- Two endpoints:
  - `/` → Returns Hello World message
  - `/data` → Returns sample JSON data
- Fully containerized using Docker

---

## 📂 Project Structure
```
fastapi-docker-api/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🧑‍💻 API Endpoints

### 1. Root Endpoint
```
GET /
```

## Response:

```
{
  "message": "Hello World"
}
```

### 2. Data Endpoint

```
GET /data
```

## Response:
```json
{
  "name": "Dinesh Kumar",
  "age": 25,
  "course": "AI/ML pg course"
}
```

## ⚙️ Prerequisites

Make sure you have installed:

Docker\
Python (optional, if running without Docker)

## 🐳 Docker Setup
1. Build Docker Image
docker build -t fastapi-app .

2. Run Docker Container
docker run -p 8000:8000 fastapi-app

3. Access the API
http://localhost:8000
http://localhost:8000/data

## 📦 Requirements
fastapi\
uvicorn

## ▶️ Run Without Docker (Optional)
pip install -r requirements.txt
uvicorn main:app --reload

## 👤 Author

Dinesh Kumar Katta

## ✅ Conclusion

This project demonstrates the basic implementation of a backend API using FastAPI and its containerization using Docker. It highlights how Docker helps in creating a consistent and portable environment for running applications across different systems.