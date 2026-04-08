import os
import json
import textwrap
from typing import List, Optional, Any, Dict

from openai import OpenAI

from environment import Action, SiliconArchitectOpenEnv
from tasks import TASKS

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_key_to_prevent_crash"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "SiliconArchitect-v1"

SYSTEM_PROMPT = """You optimize low-level kernel launch parameters for a virtual accelerator.
Return only compact JSON with keys: block_size_x, block_size_y, unroll_factor, use_shared_memory.
Respect hardware limits exactly and tailor the parameters to the task description.
Reply with absolutely NO markdown formatting or explanation, just the raw JSON."""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, step: int, observation: Dict[str, Any], last_reward: float, last_error: Optional[str]) -> tuple[Action, str]:
    user_prompt = f"Step: {step}\nLast Reward: {last_reward}\nLast Error: {last_error}\nObservation: {json.dumps(observation)}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        payload = json.loads(text)
        action = Action(**payload)
        action_str = json.dumps(payload, separators=(',', ':'))
        return action, action_str
    except Exception as exc:
        action_str = f"{{fallback_due_to_err:\"{str(exc)}\"}}"
        return Action(block_size_x=32, block_size_y=1, unroll_factor=1, use_shared_memory=False), action_str


def run_episode(client: OpenAI, task_id: str) -> None:
    env = SiliconArchitectOpenEnv(task_id=task_id)
    observation = env.reset()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    last_reward = 0.0
    last_error = None
    
    try:
        for step in range(1, env.max_steps + 1):
            obs_payload = observation.model_dump() if hasattr(observation, "model_dump") else observation.dict()
            action, action_str = get_model_action(client, step, obs_payload, last_reward, last_error)
            
            try:
                result = env.step(action)
                reward_val = result["reward"].total
                done = result["done"]
                error = None
                
                metrics = result["info"]["metrics"]
                if metrics.get("invalid"):
                    error = "invalid configuration"
                if metrics.get("oom"):
                    error = f"{error + ', ' if error else ''}oom"
                        
            except Exception as e:
                reward_val = 0.0
                done = True
                error = str(e)
            
            action_str = action_str.replace('\\n', ' ').replace('\\r', '')
            if error:
                error = error.replace('\\n', ' ').replace('\\r', '')
                
            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)
            
            rewards.append(reward_val)
            steps_taken = step
            last_reward = reward_val
            last_error = error
            
            if 'result' in locals() and isinstance(result, dict) and "observation" in result:
               observation = result["observation"]
               
            if done:
                break
                
        score = env.state().best_score
        success = score > 0.5
        
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    task_filter = os.getenv("SILICON_ARCHITECT_TASK")
    tasks_to_run = [task_filter] if task_filter and task_filter in TASKS else list(TASKS.keys())
    
    for task_id in tasks_to_run:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
