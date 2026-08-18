# MathLearningAgent

A personalized high-school mathematics learning agent for wrong-problem diagnosis, targeted exercise generation, solution verification, and learning analytics.

**Status:** Early Development

## Phase 1

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Fill in the LLM_* values in .env before running the next two commands.
python -m scripts.test_llm
python -m math_learning_agent.graph.demo_graph
python -m pytest -q
```
