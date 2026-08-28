"""FastAPI アプリ本体。

エンドポイント:
  GET  /                デモUI
  GET  /api/docs        同梱ドキュメントの一覧
  POST /api/ask         質問に回答（検索 + Claude生成）
  POST /api/upload-pdf  PDFをチャンク化して返す（サーバーには保存しない）

設計メモ: サーバーレス環境（Vercel）ではインスタンスをまたいで状態を保持できないため、
アップロードされたPDFのチャンクはクライアント側に返し、質問のたびにリクエストに
載せてもらうステートレス設計にしている。
"""

from __future__ import annotations

import io
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from .chunking import Chunk, chunk_markdown, chunk_pdf_pages
from .rag import generate_answer
from .search import BM25Index

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent / "data" / "docs"

MAX_PDF_BYTES = 2 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_EXTRA_CHUNKS = 400
TOP_K = 6

RATE_LIMIT_PER_DAY = 30
_request_log: dict[str, list[float]] = defaultdict(list)

app = FastAPI(title="PDF参照 RAGチャット デモ", docs_url=None, redoc_url=None)

# 同梱ドキュメントは起動時に一度だけインデックス化する
_builtin_chunks: list[Chunk] = []
for path in sorted(DOCS_DIR.glob("*.md")):
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
    _builtin_chunks.extend(chunk_markdown(title, text))
_builtin_index = BM25Index(_builtin_chunks)


class ExtraChunk(BaseModel):
    doc: str = Field(max_length=200)
    location: str = Field(max_length=100)
    text: str = Field(max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    extra_chunks: list[ExtraChunk] = Field(default_factory=list, max_length=MAX_EXTRA_CHUNKS)


def _check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    _request_log[ip] = [t for t in _request_log[ip] if now - t < 86400]
    if len(_request_log[ip]) >= RATE_LIMIT_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail="デモ版の利用上限（1日30回）に達しました。日を改めてお試しください。",
        )
    _request_log[ip].append(now)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/docs")
def list_docs() -> JSONResponse:
    docs: dict[str, int] = {}
    for chunk in _builtin_chunks:
        docs[chunk.doc] = docs.get(chunk.doc, 0) + 1
    return JSONResponse([
        {"doc": name, "chunks": count} for name, count in docs.items()
    ])


@app.post("/api/ask")
def ask(request: Request, body: AskRequest) -> JSONResponse:
    _check_rate_limit(request)

    if body.extra_chunks:
        extras = [
            Chunk(doc=c.doc, location=c.location, text=c.text)
            for c in body.extra_chunks
        ]
        index = BM25Index(_builtin_chunks + extras)
    else:
        index = _builtin_index

    hits = index.search(body.question, top_k=TOP_K)
    answer, ai_used = generate_answer(body.question, hits)

    return JSONResponse({
        "answer": answer,
        "ai_used": ai_used,
        "sources": [
            {
                "ref": i + 1,
                "doc": chunk.doc,
                "location": chunk.location,
                "snippet": chunk.text[:160],
                "score": round(score, 2),
            }
            for i, (chunk, score) in enumerate(hits)
        ],
    })


@app.post("/api/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile) -> JSONResponse:
    _check_rate_limit(request)
    raw = await file.read()
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="PDFのサイズ上限は2MBです")
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="PDFとして読み込めませんでした") from exc
    if len(reader.pages) > MAX_PDF_PAGES:
        raise HTTPException(status_code=400, detail=f"ページ数の上限は{MAX_PDF_PAGES}ページです")

    pages = [(page.extract_text() or "") for page in reader.pages]
    doc_name = (file.filename or "アップロードPDF").rsplit(".", 1)[0]
    chunks = chunk_pdf_pages(doc_name, pages)[:MAX_EXTRA_CHUNKS]
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="テキストを抽出できませんでした（画像スキャンのPDFは非対応です）",
        )
    return JSONResponse({
        "doc": doc_name,
        "pages": len(reader.pages),
        "chunks": [c.to_dict() for c in chunks],
    })
