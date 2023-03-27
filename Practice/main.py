import dill

import pandas as pd

import json

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: float
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: float


class Prediction(BaseModel):
    id: int
    price: int
    Result: str


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'id': form.id,
        'price': form.price,
        'Result': y[0]
    }




