import textwrap
import pandas as pd
import os
from io import BytesIO
from fpdf import FPDF
from google import genai

def build_report_prompt(cluster_means: pd.DataFrame) -> str:
    """
    クラスタ別平均RFMから、LLMに渡すプロンプト文字列を生成する。
    """
    # DataFrameをテキストに整形
    table_text = cluster_means.round(2).to_markdown()

    prompt = f"""
    あなたは、データドリブンマーケティングに詳しいアナリストです。
    以下は、RFM分析とK-Meansクラスタリングの結果として得られた
    「クラスタ別の平均値」です。

    テーブル:
    {table_text}

    各列の意味:
    - Recency: 最終購入からの日数（小さいほど最近購入している）
    - Frequency: 購入回数
    - Monetary: 総購入金額

    上記のクラスタ別指標をもとに、以下の観点で日本語のレポートを書いてください。

    1. 各クラスタの特徴（どのような顧客層か）をわかりやすく説明
    2. 重要なインサイト（例：売上を支えているのはどの層か、離脱傾向の層はどこか）
    3. 各クラスタごとに推奨されるマーケティング施策（3〜5個程度）
    4. 全体としての戦略提案（どの層に優先的に投資すべきかなど）

    出力フォーマット:
    - 見出し（例: 「1. クラスタ概要」「2. 各クラスタの特徴」など）
    - 箇条書きも活用し、マーケ担当者がすぐ読める形にする
    - です・ます調で書く
    """

    # 不要なインデントを削除
    return textwrap.dedent(prompt)


def generate_llm_report(cluster_means: pd.DataFrame, model_name: str = "gpt-4o-mini") -> str:
    """
    クラスタ別平均RFMからLLMレポートを生成して文字列として返す。
    """
    prompt = build_report_prompt(cluster_means)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )

    return response.text


def to_pdf_bytes(text: str) -> bytes:
    """
    レポート文字列をPDFバイナリに変換して返す。
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in text.splitlines():
        pdf.multi_cell(0, 10, txt=line)

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
