# src/web_server.py

from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage

from sqlalchemy import text
from .db import SessionLocal  # 使用我們在 db.py 建好的 SessionLocal

from .rag_graph import build_graph


# ===== 建立 LangGraph App =====
graph_app = build_graph()

# ===== FastAPI App =====
app = FastAPI()

# 靜態檔案（前端）位置：專案根目錄的 web/
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"


# ===== Pydantic models =====

class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    hospital: str
    messages: List[ChatMessageIn]


class ChatResponse(BaseModel):
    messages: List[ChatMessageIn]


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class AdminLoginResponse(BaseModel):
    success: bool
    message: str


# Issue 列表用的輸出資料結構（給 /api/admin/issues 使用）
class IssueItem(BaseModel):
    issue_id: int
    hospital_name: Optional[str] = None
    category_name: Optional[str] = None
    title: str
    status_code: Optional[str] = None
    priority_code: Optional[str] = None
    assignee_name: Optional[str] = None
    # 前端只要顯示即可，用字串最簡單
    created_at: str


class IssueListResponse(BaseModel):
    items: List[IssueItem]


# Issue 詳細頁用的輸出資料結構（給 /api/admin/issues/{issue_id} 使用）
class IssueDetail(BaseModel):
    issue_id: int
    title: str
    hospital_name: Optional[str] = None
    category_name: Optional[str] = None
    status_code: Optional[str] = None
    priority_code: Optional[str] = None
    assignee_name: Optional[str] = None
    description: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None


class IssueDetailResponse(BaseModel):
    issue: IssueDetail


# 使用者通報 Issue 用的輸入資料結構（給 /api/report_issue 使用）
class ReportIssueRequest(BaseModel):
    hospital: str      # 前端選單顯示的院區名稱，例如「萬芳醫院」
    title: str         # Issue 標題（例如「登入沒有畫面（使用者通報）」）
    description: str   # LLM 產生的摘要文字


# ===== 小工具：前端 JSON <-> LangChain Messages 轉換 =====

def lc_messages_from_client(msgs: List[ChatMessageIn]) -> List[AnyMessage]:
    lc_msgs: List[AnyMessage] = []
    for m in msgs:
        if m.role == "user":
            lc_msgs.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_msgs.append(AIMessage(content=m.content))
    return lc_msgs


def client_messages_from_lc(lc_msgs: List[AnyMessage]) -> List[ChatMessageIn]:
    result: List[ChatMessageIn] = []
    for m in lc_msgs:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            # system 等角色就先不傳回前端
            continue
        result.append(ChatMessageIn(role=role, content=m.content))
    return result


# ===== 小工具：驗證管理者登入（查 MySQL 的 users 表） =====

def authenticate_admin(username: str, password: str) -> bool:
    """
    目前先用「明碼比對」：
    - 去 users 表找 USERNAME
    - 確認 IS_ACTIVE = 1
    - 比對 PASSWORD_HASH 是否等於傳入的 password

    之後如果要改成 bcrypt 雜湊，只要改這裡的比對方式即可。
    """
    db = SessionLocal()
    try:
        sql = text(
            """
            SELECT USER_ID, PASSWORD_HASH, IS_ACTIVE
            FROM users
            WHERE USERNAME = :u
            """
        )
        row = db.execute(sql, {"u": username}).mappings().first()

        # 找不到帳號
        if not row:
            return False

        # 帳號被停用
        if row["IS_ACTIVE"] != 1:
            return False

        # 目前先直接明碼比對
        if row["PASSWORD_HASH"] != password:
            return False

        return True
    finally:
        db.close()


# ===== API：聊天主入口 =====

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    lc_history = lc_messages_from_client(request.messages)

    state = {
        "messages": lc_history,
        "hospital": request.hospital,
    }

    # 呼叫 LangGraph，取得新的 state（會多一則 AI 回覆）
    new_state = graph_app.invoke(state)
    new_lc_messages = new_state["messages"]

    client_msgs = client_messages_from_lc(new_lc_messages)
    return ChatResponse(messages=client_msgs)


# ===== API：使用者通報 Issue =====

@app.post("/api/report_issue")
async def report_issue(req: ReportIssueRequest):
    """
    使用者在聊天頁按下「我要通報 Issue」時呼叫。
    會：
    1. 依院區名稱找 hospitals 表的 HOSPITAL_ID
    2. 寫入 issues 表（先給預設狀態 open / normal）
    之後如果要自動分類 / 指派負責人，可以再擴充這支 API。
    """
    db = SessionLocal()
    try:
        # 1. 依院區名稱找 HOSPITAL_ID
        sql_hos = text(
            "SELECT HOSPITAL_ID FROM hospitals WHERE HOSPITAL_NAME = :h"
        )
        hos_row = db.execute(sql_hos, {"h": req.hospital}).mappings().first()
        if not hos_row:
            return {"success": False, "message": "找不到對應院區"}

        hospital_id = hos_row["HOSPITAL_ID"]

        # 2. 寫入 issues 表
        sql_ins = text(
            """
            INSERT INTO issues (
                HOSPITAL_ID,
                TITLE,
                DESCRIPTION,
                STATUS_CODE,
                PRIORITY_CODE,
                CREATED_AT,
                UPDATED_AT
            )
            VALUES (
                :hid,
                :title,
                :desc,
                'open',          -- 先固定為 open
                'normal',        -- 先固定為 normal
                NOW(),
                NOW()
            )
            """
        )
        result = db.execute(sql_ins, {
            "hid": hospital_id,
            "title": req.title,
            "desc": req.description,
        })
        db.commit()

        new_id = result.lastrowid if hasattr(result, "lastrowid") else None

        return {
            "success": True,
            "message": "Issue 建立成功",
            "issue_id": new_id,
        }
    except Exception as e:
        print("report_issue error:", e)
        return {
            "success": False,
            "message": "後端錯誤，請稍後再試",
        }
    finally:
        db.close()


# ===== API：管理者登入（查資料庫） =====

@app.post("/api/admin/login", response_model=AdminLoginResponse)
async def admin_login(request: AdminLoginRequest) -> AdminLoginResponse:
    ok = authenticate_admin(request.username, request.password)

    if not ok:
        return AdminLoginResponse(
            success=False,
            message="帳號或密碼錯誤，或帳號已停用",
        )

    return AdminLoginResponse(success=True, message="登入成功")


# ===== API：Issue 列表（給 admin_issues.html 用） =====

@app.get("/api/admin/issues", response_model=IssueListResponse)
async def list_issues() -> IssueListResponse:
    """
    從 issues + hospitals + issue_category + users 撈資料，
    做成簡單列表給前端表格顯示。
    """
    db = SessionLocal()
    try:
        sql = text(
            """
            SELECT
                i.ISSUE_ID,
                h.HOSPITAL_NAME,
                c.CATEGORY_NAME,
                i.TITLE,
                i.STATUS_CODE,
                i.PRIORITY_CODE,
                u.DISPLAY_NAME AS ASSIGNEE_NAME,
                i.CREATED_AT
            FROM issues i
            LEFT JOIN hospitals h
                ON i.HOSPITAL_ID = h.HOSPITAL_ID
            LEFT JOIN issue_category c
                ON i.CATEGORY_ID = c.CATEGORY_ID
            LEFT JOIN users u
                ON i.ASSIGNEE_USER_ID = u.USER_ID
            ORDER BY i.CREATED_AT DESC
            """
        )
        rows = db.execute(sql).mappings().all()

        items: List[IssueItem] = []
        for r in rows:
            created_at = r["CREATED_AT"]
            if created_at is not None:
                created_str = created_at.isoformat(sep=" ", timespec="seconds")
            else:
                created_str = ""

            items.append(
                IssueItem(
                    issue_id=r["ISSUE_ID"],
                    hospital_name=r["HOSPITAL_NAME"],
                    category_name=r["CATEGORY_NAME"],
                    title=r["TITLE"],
                    status_code=r["STATUS_CODE"],
                    priority_code=r["PRIORITY_CODE"],
                    assignee_name=r["ASSIGNEE_NAME"],
                    created_at=created_str,
                )
            )

        return IssueListResponse(items=items)
    finally:
        db.close()


# ===== API：Issue 詳細資料（給 admin_issue_detail.html 用） =====

@app.get("/api/admin/issues/{issue_id}", response_model=IssueDetailResponse)
async def get_issue_detail(issue_id: int) -> IssueDetailResponse:
    """
    取得單一 Issue 的詳細資訊（包含描述）。
    目前先從 issues + hospitals + issue_category + users 撈資料，
    之後如果有對話摘要 / 處理歷程，再繼續擴充這支 API。
    """
    db = SessionLocal()
    try:
        sql = text(
            """
            SELECT
                i.ISSUE_ID,
                i.TITLE,
                h.HOSPITAL_NAME,
                c.CATEGORY_NAME,
                i.STATUS_CODE,
                i.PRIORITY_CODE,
                u.DISPLAY_NAME AS ASSIGNEE_NAME,
                i.DESCRIPTION,
                i.CREATED_AT,
                i.UPDATED_AT
            FROM issues i
            LEFT JOIN hospitals h
                ON i.HOSPITAL_ID = h.HOSPITAL_ID
            LEFT JOIN issue_category c
                ON i.CATEGORY_ID = c.CATEGORY_ID
            LEFT JOIN users u
                ON i.ASSIGNEE_USER_ID = u.USER_ID
            WHERE i.ISSUE_ID = :id
            """
        )
        row = db.execute(sql, {"id": issue_id}).mappings().first()

        if not row:
            raise HTTPException(status_code=404, detail="Issue not found")

        created_at = row["CREATED_AT"]
        if created_at is not None:
            created_str = created_at.isoformat(sep=" ", timespec="seconds")
        else:
            created_str = ""

        updated_at = row["UPDATED_AT"]
        if updated_at is not None:
            updated_str = updated_at.isoformat(sep=" ", timespec="seconds")
        else:
            updated_str = ""

        issue = IssueDetail(
            issue_id=row["ISSUE_ID"],
            title=row["TITLE"],
            hospital_name=row["HOSPITAL_NAME"],
            category_name=row["CATEGORY_NAME"],
            status_code=row["STATUS_CODE"],
            priority_code=row["PRIORITY_CODE"],
            assignee_name=row["ASSIGNEE_NAME"],
            description=row["DESCRIPTION"],
            created_at=created_str,
            updated_at=updated_str,
        )
        return IssueDetailResponse(issue=issue)
    finally:
        db.close()


# ===== 首頁與靜態頁面 =====

@app.get("/", response_class=HTMLResponse)
async def root():
    # 首頁：選擇 一般使用者 / 管理者
    return FileResponse(WEB_DIR / "index.html")


@app.get("/user", response_class=HTMLResponse)
async def user_page():
    # 一般使用者聊天頁
    return FileResponse(WEB_DIR / "user.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    # 管理者登入頁
    return FileResponse(WEB_DIR / "admin.html")


@app.get("/admin/issues", response_class=HTMLResponse)
async def admin_issues_page():
    # 管理者 Issue 列表頁
    return FileResponse(WEB_DIR / "admin_issues.html")


@app.get("/admin/issues/{issue_id}", response_class=HTMLResponse)
async def admin_issue_detail_page(issue_id: int):
    """
    管理者 Issue 詳細頁。
    issue_id 先不用管，在前端 admin_issue_detail.html 會用 JS 讀網址再去 call API。
    """
    return FileResponse(WEB_DIR / "admin_issue_detail.html")


# 靜態資源（CSS / JS / 圖片）
# 注意：如果你的 HTML 裡是寫 href="/static/style.css"
# 就會由這裡提供檔案，路徑是 web/style.css
app.mount("/static", StaticFiles(directory=str(WEB_DIR), html=False), name="static")
