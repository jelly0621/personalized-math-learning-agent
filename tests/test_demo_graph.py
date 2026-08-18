"""Offline tests for the demo graph's deterministic node."""

from math_learning_agent.graph.demo_graph import prepare_prompt


def test_prepare_prompt_uses_user_input() -> None:
    state = {
        "user_input": "用一句话解释什么是函数。",
        "prompt": "",
        "llm_response": "",
    }

    update = prepare_prompt(state)

    assert state["user_input"] in update["prompt"]
    assert update["prompt"].startswith("请清晰、简洁地回答")
    assert state["llm_response"] == ""
