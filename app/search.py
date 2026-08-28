"""検索（リトリーバル）部分。

依存ライブラリなしの BM25 実装。日本語は形態素解析の代わりに文字バイグラムで
トークン化する（辞書不要でサーバーレス環境でも軽量に動く、実務でも定番の手法）。

実案件で大規模・高精度が必要な場合はベクトル検索やハイブリッド検索
（例: Azure AI Search）に置き換える前提の、デモ用ミニマル実装。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .chunking import Chunk

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """英数字は単語単位、日本語（CJK）は文字バイグラムでトークン化する。"""
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9]+|[぀-ヿ一-鿿゠-ヿ々ー]+", text):
        word = match.group(0)
        if re.match(r"[a-z0-9]", word):
            tokens.append(word)
        else:
            if len(word) == 1:
                tokens.append(word)
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(f"{c.location} {c.text}") for c in chunks]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.avg_len = (
            sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens else 0.0
        )
        # 各トークンが何チャンクに出現するか
        self.df: Counter[str] = Counter()
        for freq in self.doc_freqs:
            self.df.update(freq.keys())

    def _idf(self, token: str) -> float:
        n = len(self.chunks)
        df = self.df.get(token, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
        query_tokens = tokenize(query)
        scores = [0.0] * len(self.chunks)
        for token in set(query_tokens):
            idf = self._idf(token)
            for i, freq in enumerate(self.doc_freqs):
                tf = freq.get(token, 0)
                if tf == 0:
                    continue
                length_norm = 1 - B + B * len(self.doc_tokens[i]) / self.avg_len
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * length_norm)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], scores[i]) for i in ranked[:top_k] if scores[i] > 0]
