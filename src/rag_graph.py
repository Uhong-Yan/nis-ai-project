# src/rag_graph.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, TypedDict, Annotated

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .config import get_llm


# ===== 路徑與設定 =====
BASE_DIR = Path(__file__).resolve().parent.parent
EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")


# ===== 院區代碼標準化 =====
def normalize_hospital_code(raw: str) -> str:
    """
    把前端傳進來的院區字串，轉成我們在 data/ 底下使用的「院區代碼」。
    例如：
      - "wanfang"、"萬芳醫院"  -> "wanfang"
    之後如果有別的院區，再往這裡加就好。
    """
    if not raw:
        return "wanfang"

    r = raw.strip().lower()
    if r in {"wanfang", "wf"}:
        return "wanfang"

    # 中文名稱也試著處理
    if "萬芳" in raw:
        return "wanfang"

    # 找不到對應時，先回落到萬芳，避免整個炸掉
    return "wanfang"


# ===== 簡單的 FAISS 檢索器（依院區載入） =====
class SimpleFaissRetriever:
    """
    給定院區代碼 (hospital_code)，從對應資料夾載入向量庫：

        data/{hospital_code}/vectorstore/faiss.index
        data/{hospital_code}/vectorstore/docs.npy

    利用 sentence-transformers 做 embedding，再從 FAISS 找最相近的 k 筆文件。
    """

    def __init__(self, hospital_code: str, k: int = 5):
        self.k = k
        self.hospital_code = hospital_code

        base_vs_dir = BASE_DIR / "data" / hospital_code / "vectorstore"

        # 讀取向量索引 & 對應的 Document 清單
        self.index = faiss.read_index(str(base_vs_dir / "faiss.index"))
        self.docs: List[Document] = np.load(
            base_vs_dir / "docs.npy", allow_pickle=True
        ).tolist()

        # 載入與 ingest 時相同的 embedding model
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        給一個 query 字串，回傳前 k 筆最相近的 Document。
        """
        # 將 query 轉成向量
        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        # 在 FAISS 中搜尋
        distances, indices = self.index.search(q_vec, self.k)
        idxs = indices[0]
        return [self.docs[i] for i in idxs]


# ===== LangGraph 狀態定義 =====
class ChatState(TypedDict):
    """
    LangGraph 的 state。
    - messages：對話歷史，使用 add_messages 讓每一步自動 append 訊息
    - hospital：院區代碼（例如 "wanfang"），由前端 / CLI 在進入對話前決定
    """
    messages: Annotated[list[AnyMessage], add_messages]
    hospital: str


# ===== 建立共用的 LLM =====
llm = get_llm()


# ===== RAG 節點：根據最後一個使用者訊息做檢索＋回答 =====
def rag_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    raw_hospital = state.get("hospital") or "wanfang"
    hospital_code = normalize_hospital_code(raw_hospital)

    # 找最後一句 HumanMessage（使用者最新問的那一句）
    last_user: HumanMessage | None = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user = m
            break

    if last_user is None:
        # 沒有使用者訊息就直接回傳原本狀態
        return {"messages": messages, "hospital": hospital_code}

    query = last_user.content

    # 1. 先嘗試 RAG 檢索，用對應院區的向量庫
    context_text = ""
    try:
        retriever = SimpleFaissRetriever(hospital_code=hospital_code, k=5)
        docs = retriever.get_relevant_documents(query)
        if docs:
            context_text = "\n\n".join(
                f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
            )
    except Exception as e:
        # 如果 vectorstore 有問題，不讓整個系統掛掉；只是沒有 context 而已
        context_text = ""
        # 你要的 demo 不需要顯示錯誤，log 在伺服器就好
        # print(f"RAG 檢索失敗：{e!r}")

    # 2. 組成系統提示＋歷史對話，丟給 LLM
    # 這裡重點：讓回覆「像真人客服」、不要一直講「文件 5」
    hospital_display = "萬芳醫院" if hospital_code == "wanfang" else hospital_code

    system_prompt = (
        "你是「NIS 小幫手」，是一位服務護理師與醫療人員的友善客服專員。"
        "請用自然、口語化、但專業且有禮貌的繁體中文回答。\n\n"
        "【角色設定】\n"
        f"- 目前服務院區：{hospital_display}。\n"
        "- 你熟悉該院區的護理資訊系統（NIS）操作與相關規範。\n"
        "- 使用者多半是忙碌的護理師或醫師，希望你回答清楚、步驟明確。\n\n"
        "【回答風格】\n"
        "1. 優先使用下方提供的「系統操作說明內容」來回答問題。\n"
        "2. 請避免說「根據文件 5」「如文件中所述」這種字眼，改用：\n"
        "   -「依照系統操作說明，可以這樣做…」\n"
        "   -「在 NIS 的設定流程中，通常會…」\n"
        "3. 回覆時可以簡短分段，先給結論，再提供 1～3 個具體步驟。\n"
        "4. 如果資料內容不足以回答，請誠實說明，並建議：\n"
        "   -「這部分系統說明沒有寫得很清楚，建議聯繫系統管理者確認。」\n"
        "   絕對不要亂編造功能或步驟。\n"
        "5. 若使用者明顯在抱怨或情緒緊張，可以先簡短同理再給建議。\n\n"
        "【系統操作說明內容（僅供你參考，不要逐字照念出標題編號）】\n"
    )

    if context_text:
        system_prompt = system_prompt + context_text
    else:
        system_prompt = (
            system_prompt
            + "（目前查不到相關操作說明，若無法回答請誠實說明，並建議使用者改由系統管理者協助。）"
        )

    system_msg = SystemMessage(content=system_prompt)

    model_input: List[AnyMessage] = [system_msg] + messages

    # 3. 呼叫 LLM 產生回覆
    ai_reply: AIMessage = llm.invoke(model_input)

    # 4. 把 AI 回覆加回 state，LangGraph 會幫我們維護多輪對話
    new_messages = messages + [ai_reply]
    return {"messages": new_messages, "hospital": hospital_code}


# ===== 建立並編譯 LangGraph =====
def build_graph():
    """
    建立一個最簡單的單節點 RAG 對話圖：
    entry -> rag -> END

    之後如果要加「判斷要不要 RAG」「切換工具」「不同院區路由」，都可以在這裡擴充更多節點與 edge。
    """
    graph = StateGraph(ChatState)

    # 只有一個節點：rag
    graph.add_node("rag", rag_node)

    # 入口 → rag → END
    graph.set_entry_point("rag")
    graph.add_edge("rag", END)

    app = graph.compile()
    return app
