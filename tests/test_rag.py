"""チャンク分割・検索・パイプラインの動作確認（APIキー不要・回答はモック）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_markdown, chunk_pdf_pages  # noqa: E402
from app.search import BM25Index, tokenize  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def _load_index() -> BM25Index:
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        chunks.extend(chunk_markdown(title, text))
    return BM25Index(chunks)


def test_tokenize():
    tokens = tokenize("CSV取り込みの上限は10,000行")
    assert "csv" in tokens
    assert "10" in tokens and "000" in tokens
    assert "取り" in tokens or "取込" in tokens or "り込" in tokens  # バイグラム


def test_chunk_markdown_sections():
    chunks = chunk_markdown("テスト", "# T\n\n## A\n\n本文A\n\n### A-1\n\n本文A1\n\n## B\n\n本文B")
    locations = [c.location for c in chunks]
    assert "A" in locations
    assert "A > A-1" in locations
    assert "B" in locations  # 下位見出しがリセットされている


def test_chunk_pdf_pages():
    chunks = chunk_pdf_pages("PDF", ["1ページ目の本文", "", "3ページ目の本文"])
    assert [c.location for c in chunks] == ["p.1", "p.3"]  # 空ページはスキップ


def test_search_hits_correct_doc():
    index = _load_index()

    hits = index.search("解約はいつまでに申請すればいいですか")
    assert hits, "検索結果が空"
    top = hits[0][0]
    assert "料金プラン" in top.location or "解約" in top.text

    hits = index.search("無断キャンセルのペナルティ")
    assert any("無断キャンセル" in c.text for c, _ in hits[:2])

    hits = index.search("APIのレート制限")
    assert any("120リクエスト" in c.text for c, _ in hits[:3])


def test_search_no_hit_returns_empty():
    index = _load_index()
    hits = index.search("量子コンピュータの誤り訂正について")
    # 無関係な質問でも何かしらのバイグラムが当たることはあるが、上位が支配的でないこと
    assert len(hits) == 0 or hits[0][1] < 10


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("all tests passed")
