"""A minimal two-node LangGraph example."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from math_learning_agent.llm import LLMClient


class DemoState(TypedDict):
    """State shared by all nodes in the demo graph."""

    user_input: str
    prompt: str
    llm_response: str


def prepare_prompt(state: DemoState) -> dict[str, str]:
    """Turn the user's input into a simple instruction for the LLM."""

    return {
        "prompt": (
            "请清晰、简洁地回答下面的问题，不要添加无关内容：\n"
            f"{state['user_input']}"
        )
    }


def call_llm(state: DemoState) -> dict[str, str]:
    """Call the configured LLM and store its text response."""

    client = LLMClient()
    response = client.invoke([{"role": "user", "content": state["prompt"]}])
    return {"llm_response": response}


def build_demo_graph():
    """Build START -> prepare_prompt -> call_llm -> END."""

    builder = StateGraph(DemoState)
    builder.add_node("prepare_prompt", prepare_prompt)
    builder.add_node("call_llm", call_llm)
    builder.add_edge(START, "prepare_prompt")
    builder.add_edge("prepare_prompt", "call_llm")
    builder.add_edge("call_llm", END)
    return builder.compile()


demo_graph = build_demo_graph()


if __name__ == "__main__":
    final_state = demo_graph.invoke(
        {
            "user_input": "用一句话解释什么是函数。",
            "prompt": "",
            "llm_response": "",
        }
    )
    print(final_state["llm_response"])
