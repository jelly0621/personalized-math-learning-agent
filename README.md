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

## Phase 2

```powershell
python -m scripts.demo_wrong_problem
```

第一次运行会在 `data/` 下创建本地 SQLite 数据库。

## Phase 3

当前流程：

Wrong Problem → Problem Understanding → Error Diagnosis → SQLite Persistence → Student Knowledge Profile

```powershell
python -m scripts.demo_error_diagnosis
```

## Phase 4

Student History → Load Student Profile → Training Planner → Personalized Training Plan

```powershell
python -m scripts.demo_training_plan
```
