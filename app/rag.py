"""回答生成（Claude API）。

検索でヒットした抜粋だけを根拠に回答させ、出典番号【n】を必ず付けさせる。
根拠がなければ「わからない」と答えさせる（ハルシネーション対策の基本形）。

ANTHROPIC_API_KEY 未設定の環境ではモック回答で動作する。
"""

from __future__ import annotations

import os

import anthropic

from .chunking import Chunk

# デモ用途のためコスト最優先で Haiku を使用
MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """\
あなたは社内ドキュメント検索アシスタントです。
ユーザーの質問に対し、与えられた「参照抜粋」だけを根拠に日本語で回答してください。

ルール:
- 回答の根拠となる箇所には必ず出典番号を【1】のように付けること
- 参照抜粋に書かれていないことは推測で補わない。根拠が見つからない場合は
  「ご提供のドキュメントからは確認できませんでした」と明確に answer すること
- 回答は簡潔に。箇条書きが適する場合は箇条書きを使う
- 数値・期限・条件は抜粋の記載を正確に引き写すこと
"""


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        blocks.append(f"【{i}】{chunk.doc} / {chunk.location}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, hits: list[tuple[Chunk, float]]) -> tuple[str, bool]:
    """回答テキストと (AIを使ったか) を返す。"""
    if not hits:
        return "関連する記述がドキュメント内に見つかりませんでした。質問の言い換えをお試しください。", False

    if not os.environ.get("ANTHROPIC_API_KEY"):
        top = hits[0][0]
        return (
            "[モック回答（APIキー未設定）] 検索は実際に動作しています。"
            f"最も関連度が高い抜粋は【1】{top.doc} / {top.location} でした。"
            "実環境ではこの抜粋を根拠に Claude が回答を生成します。",
            False,
        )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"# 参照抜粋\n\n{_format_context(hits)}\n\n"
                f"# 質問\n\n{question}"
            ),
        }],
    )
    answer = "".join(block.text for block in response.content if block.type == "text")
    return answer.strip(), True
