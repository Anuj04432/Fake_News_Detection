from fastapi import FastAPI
# from pydantic import BaseModel

app = FastAPI()

app.get("/")
def home_page():
    return "Welcome to the fake news detection app"