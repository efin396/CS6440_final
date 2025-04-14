import uvicorn
from api import app  # Replace with the actual module where your FastAPI app is defined

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)