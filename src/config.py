import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# 日後如果要用 LangChain 的 Document / Messages
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()  # 讀 .env


def get_llm():
    """
    回傳接到 OpenRouter 的 ChatOpenAI，
    使用 meta-llama/llama-3.3-70b-instruct:free。
    """
    api_key = os.environ["OPENROUTER_API_KEY"]

    llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        # 這兩個 header 是 OpenRouter 推薦，可以讓你的 app 出現在排行榜上（非必填）
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "nis-ai-project"),
        },
        temperature=0.1,
    )
    return llm
