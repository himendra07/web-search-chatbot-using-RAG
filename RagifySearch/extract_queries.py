from pydantic import BaseModel
from config import MODEL_NAME, PROMPT
from ollama import chat
from datetime import datetime
import json

class Queries(BaseModel):
    queries: list[str]

def extract_queries(query: str, model: str = MODEL_NAME) -> list[str]:
    """
    Extracts a list of queries from the given user query using Ollama chat.
    Returns a list of strings; never returns None.
    """

    if not model:
        model = MODEL_NAME
    model = str(model)

    prompt_text = PROMPT.format(date=datetime.today().strftime("%Y-%m-%d"), input_query=query)

    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": 0.1},
            format=Queries.model_json_schema()
        )
        content = response.get("message", {}).get("content", "")
    except Exception as e:
        try:
            response = chat(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                options={"temperature": 0.1},
            )
            content = response.get("message", {}).get("content", "")
        except Exception as inner:
            print("extract_queries: chat request failed:", inner)
            return [query]

    if not content:
        return [query]

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list):
            return [str(q) for q in parsed["queries"]]
        if isinstance(parsed, list):
            return [str(q) for q in parsed]
    except Exception:
        pass
        
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines:
        return lines
    return [query]
