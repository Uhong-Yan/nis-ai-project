# src/cli_chat.py
from langchain_core.messages import HumanMessage
from .rag_graph import build_graph


def main():
    app = build_graph()

    # ---- 選擇院區（之後 Web 版會改成由前端 dropdown 傳進來）----
    print("請選擇院區：")
    print("1) wanfang（萬芳醫院）")
    choice = input("輸入編號（預設 1）：").strip() or "1"

    if choice == "1":
        hospital_code = "wanfang"
    else:
        # 目前先只做萬芳，其他輸入一律當萬芳
        hospital_code = "wanfang"

    # 初始 state：沒有歷史訊息，但先記好院區
    state = {
        "messages": [],
        "hospital": hospital_code,
    }

    print(f"\nNIS AI 客服 (RAG + LangGraph) - 院區：{hospital_code}。輸入 exit 離開。")

    # ---- 對話迴圈 ----
    while True:
        user = input("你：").strip()
        if user.lower() in {"exit", "quit"}:
            break

        # 把新的 HumanMessage 加到 messages 裡，連同 hospital 一起丟進 graph
        state = app.invoke(
            {
                "messages": state["messages"] + [HumanMessage(content=user)],
                "hospital": state["hospital"],
            }
        )

        # 最新一則一定是 AI 回覆
        ai_msg = state["messages"][-1]
        print(f"AI：{ai_msg.content}\n")


if __name__ == "__main__":
    main()
