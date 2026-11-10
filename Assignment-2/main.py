from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/data")
def get_data():
    return {
        "name": "Dinesh Kumar",
        "age": 25,
        "course": "AI/ML pg course"
    }