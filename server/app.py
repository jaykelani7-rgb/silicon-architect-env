from fastapi import FastAPI, Body
from typing import Dict, Any
import uvicorn
import sys
import os

# Ensure the root directory is in the python path to import environment.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment import Action, SiliconArchitectOpenEnv

app = FastAPI()
env = SiliconArchitectOpenEnv()

@app.post("/reset")
def reset(request: Dict[str, Any] = Body(default={})):
    obs = env.reset()
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "reward": 0.0,
        "done": False,
    }

@app.post("/step")
def step(request: Dict[str, Any]):
    action_data = request.get("action", {})
    action = Action(**action_data)
    result = env.step(action)
    
    return {
        "observation": result["observation"].model_dump() if hasattr(result["observation"], "model_dump") else result["observation"].dict(),
        "reward": result["reward"].total,
        "done": result["done"],
    }

@app.get("/state")
def state():
    st = env.state()
    return st.model_dump() if hasattr(st, "model_dump") else st.dict()

@app.get("/health")
def health():
    return {"status": "healthy"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
