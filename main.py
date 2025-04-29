from fastapi import FastAPI
from pydantic import BaseModel
import catboost


app = FastAPI()

MODEL_PATH = "/workspaces/ifortex/models/catboost_model.cbm"
model = catboost.CatBoostClassifier()
model.load_model(MODEL_PATH)


class PredictRequest(BaseModel):
    Test_1: int
    Test_2: int
    Test_3: int
    Sleep_Before_Exam: str
    Mood: str
    Energy_Drink: str
    Attendance: str
    Preparation_Time: str


@app.get("/ping")
async def ping():
    return {"message": "200 OK"}


@app.post("/predict")
async def predict(request: PredictRequest):
    # prepare features in the order expected by the model
    features = [
        request.Test_1,
        request.Test_2,
        request.Test_3,
        request.Sleep_Before_Exam,
        request.Mood,
        request.Energy_Drink,
        request.Attendance,
        request.Preparation_Time,
    ]
    # CatBoost expects a 2D array for prediction
    prediction = model.predict([features])[0]

    return {"prediction": int(prediction)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
