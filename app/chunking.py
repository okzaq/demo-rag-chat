"""ドキュメントのチャンク分割。

RAGの回答品質はチャンク設計でほぼ決まるため、単純な固定長分割ではなく
「見出し構造で区切ってから、長すぎるものだけ重なり付きで分割する」方式を採る。
- Markdown: 見出し（## / ###）単位でセクション化
- PDF: ページ単位で取り出し、ページ番号を出典として保持
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_CHUNK_CHARS = 700   # これを超えるセクションは分割する
OVERLAP_CHARS = 120     # 分割時の重なり（文脈の途切れを防ぐ）


@dataclass
class Chunk:
    doc: str        # ドキュメント名
    location: str   # 出典表示（セクション名 or ページ番号）
    text: str
    chunk_id: int = field(default=0)

    def to_dict(self) -> dict:
        return {"doc": self.doc, "location": self.location, "text": self.text}


def _split_long(text: str) -> list[str]:
    """長文を文境界を優先しつつ重なり付きで分割する。"""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    sentences = re.split(r"(?<=[。！？\n])", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > MAX_CHUNK_CHARS and current:
            parts.append(current)
            current = current[-OVERLAP_CHARS:] + sentence
        else:
            current += sentence
    if current.strip():
        parts.append(current)
    return parts


def chunk_markdown(doc_name: str, text: str) -> list[Chunk]:
    """Markdownを見出し単位でセクション化してチャンクにする。"""
    chunks: list[Chunk] = []
    heading_path: dict[int, str] = {}
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        location = " > ".join(
            heading_path[level] for level in sorted(heading_path) if level > 1
        ) or heading_path.get(1, "")
        for part in _split_long(body):
            chunks.append(Chunk(doc=doc_name, location=location, text=part.strip()))

    for line in text.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+)", line)
        if match:
            flush()
            level = len(match.group(1))
            heading_path[level] = match.group(2).strip()
            # 下位レベルの見出しをリセット
            for deeper in [lv for lv in heading_path if lv > level]:
                del heading_path[deeper]
        else:
            buffer.append(line)
    flush()

    for i, chunk in enumerate(chunks):
        chunk.chunk_id = i
    return chunks


def chunk_pdf_pages(doc_name: str, pages: list[str]) -> list[Chunk]:
    """PDFのページテキスト列をチャンクにする。出典はページ番号。"""
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        cleaned = re.sub(r"[ \t]+", " ", page_text).strip()
        if not cleaned:
            continue
        for part in _split_long(cleaned):
            chunks.append(Chunk(doc=doc_name, location=f"p.{page_no}", text=part.strip()))
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = i
    return chunks
