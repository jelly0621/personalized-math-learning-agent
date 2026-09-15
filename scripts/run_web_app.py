"""Start the local MathLearningAgent Web MVP."""

import os

import uvicorn


def get_server_address() -> tuple[str, int]:
    """Resolve local defaults and Railway-provided host/port settings."""

    port_text = os.getenv("PORT", "").strip()
    host = os.getenv("HOST", "").strip()
    if not host:
        host = "0.0.0.0" if port_text else "127.0.0.1"

    if not port_text:
        return host, 8000
    try:
        port = int(port_text)
    except ValueError as exc:
        raise RuntimeError("PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("PORT must be between 1 and 65535.")
    return host, port


def main() -> None:
    host, port = get_server_address()
    uvicorn.run(
        "math_learning_agent.web.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
