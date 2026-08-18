"""Run a minimal live LLM connection check."""

from math_learning_agent.llm import LLMClient


def main() -> None:
    try:
        client = LLMClient()
        result = client.invoke(
            [
                {
                    "role": "user",
                    "content": "请只回复：LLM connection successful",
                }
            ]
        )
        print(result)
    except Exception as exc:
        print(f"LLM connection failed: {exc}")


if __name__ == "__main__":
    main()
