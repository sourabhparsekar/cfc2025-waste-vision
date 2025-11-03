# groq_compound_search_tool.py
from ibm_watsonx_orchestrate.agent_builder.tools import tool
from groq import Groq
import json
import os 
from dotenv import load_dotenv
from ibm_watsonx_orchestrate.run import connections
from ibm_watsonx_orchestrate.run.connections import ConnectionType

load_dotenv()

@tool(expected_credentials=[{'app_id': 'groq_search', 'type': ConnectionType.BEARER_TOKEN}])
def groq_compound_search(query: str) -> dict:
    """Performs a web search using Groq Compound (groq/compound-mini) and returns a structured JSON summary.

    Args:
        query (str): The search query or question.

    Returns:
        dict: {
            "query": <original query>,
            "summary": <one paragraph summary>,
            "sources": [<list of source URLs>]
        }
    """
    api_key = os.getenv("GROQ_API_KEY")
    if connections.get_connection_type("groq_search") == ConnectionType.BEARER_TOKEN:
        conn = connections.bearer_token('groq_search')
        api_key = conn.token

    client = Groq(api_key=api_key, default_headers={"Groq-Model-Version": "latest"})

    # Define system instruction for structured JSON output
    system_prompt = (
        "You are a research assistant with access to web_search and visit_website tools. "
        "Perform a search for the given query, read a few sources, and summarize findings in a single paragraph. "
        "Return ONLY a valid JSON with fields: query, summary, sources (list of URLs)."
    )

    try:
        completion = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_completion_tokens=512,
            top_p=1,
            compound_custom={
                "tools": {"enabled_tools": ["web_search", "visit_website"]}
            },
            stream=False,
        )

        message = completion.choices[0].message.content.strip()

        # Parse model output into structured JSON
        try:
            result = json.loads(message)
            return {
                "query": result.get("query", query),
                "summary": result.get("summary", ""),
                "sources": result.get("sources", []),
            }
        except json.JSONDecodeError:
            return {
                "query": query,
                "summary": message,
                "sources": [],
            }

    except Exception as e:
        return {
            "query": query,
            "summary": f"GroqCompoundSearchTool failed: {e}",
            "sources": [],
        }
