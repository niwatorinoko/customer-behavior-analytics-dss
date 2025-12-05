import textwrap
import pandas as pd
import os
from google import genai
import tempfile
import markdown


# ======================================================
# 🧠 プロンプト構築ロジック
# ======================================================
def build_report_prompt(data_summary, mode: str = "customer") -> str:
    """
    Generates an LLM prompt according to the report type.
    mode: "customer", "product", or "combined"
    """
    def df_to_text(df_or_dict):
        if isinstance(df_or_dict, pd.DataFrame):
            return df_or_dict.round(2).to_markdown(index=False)
        elif isinstance(df_or_dict, dict):
            return "\n".join(f"- {k}: {v}" for k, v in df_or_dict.items())
        else:
            return str(df_or_dict)

    if mode == "customer":
        table_text = df_to_text(data_summary["rfm"])
        prompt = f"""
        You are a data-driven marketing analyst.
        Below is the summary table of customer segmentation based on
        RFM analysis and K-Means clustering.

        Table:
        {table_text}

        Column definitions:
        - Recency: Days since the last purchase (lower means more recent)
        - Frequency: Number of purchases
        - Monetary: Total purchase amount

        Please write a professional business report in English addressing the following points:

        1. Describe the characteristics of each cluster (what kind of customer group it represents)
        2. Highlight key insights (e.g., which clusters drive revenue, which show signs of churn)
        3. Recommend 3–5 marketing actions for each cluster
        4. Suggest an overall strategic focus (which clusters deserve priority investment)

        Output format:
        - Use clear section headings such as “1. Cluster Overview”, “2. Insights”, etc.
        - Use bullet points for readability
        - Write in concise, natural business English
        """

    elif mode == "product":
        table_text = df_to_text(data_summary["forecast"])
        prompt = f"""
        You are a retail data analyst.
        Below is a summary of product sales performance and forecast results.

        Table:
        {table_text}

        Column definitions (example):
        - Product: Product name
        - SalesCount: Number of units sold
        - MAE / RMSE / R²: Forecast accuracy metrics

        Please write a concise English report covering the following:

        1. Summarize trends in best-selling and underperforming products
        2. Identify categories or products with potential growth in demand
        3. Discuss underperforming products and improvement opportunities (pricing, promotion, inventory, etc.)
        4. Propose short- and mid-term sales strategies

        Output format:
        - Use section headings such as “1. Sales Trends”, “2. Demand Forecast”, “3. Strategic Recommendations”
        - Focus on clear bullet points
        - Use concise, professional English
        """

    elif mode == "combined":
        rfm_text = df_to_text(data_summary["rfm"])
        forecast_text = df_to_text(data_summary["forecast"])
        prompt = f"""
        You are an expert in data-driven marketing analytics.
        Below are two datasets: one for customer segmentation (RFM clustering)
        and another for product sales forecasting.

        [Customer Segment Summary]
        {rfm_text}

        [Product Sales Forecast Summary]
        {forecast_text}

        Please write an integrated English marketing report addressing:

        1. The relationship between customer segments and product sales patterns
           (e.g., which high-value customers purchase which product categories)
        2. Marketing strategies tailored to each customer segment
        3. Key priorities for upcoming campaigns (cross-sell, up-sell, inventory optimization)
        4. A holistic business strategy to maximize revenue growth

        Output format:
        - Use structured sections and bullet points
        - Write in natural, professional business English
        - The tone should resemble an executive-level strategic report
        """
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return textwrap.dedent(prompt)

# ======================================================
# 🧠 LLM呼び出しロジック
# ======================================================

def generate_llm_report(data_summary, mode: str = "customer") -> str:
    """
    LLMを使ってマーケティングレポートを生成する。
    """
    prompt = build_report_prompt(data_summary, mode)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        # APIエラーをStreamlit側でハンドルしやすいように明示
        raise RuntimeError(f"LLMレポート生成中にエラーが発生しました: {e}")


# ======================================================
# 📄 PDF出力機能
# ======================================================
from weasyprint import HTML

def export_report_to_pdf(report_md: str, title: str = "Marketing Report") -> str:
    """
    Converts Markdown to HTML and exports as PDF using WeasyPrint.
    Works on ARM (Apple Silicon) and Linux x86_64 alike.
    """
    html_content = markdown.markdown(
        report_md, extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    html_full = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'DejaVu Sans', sans-serif;
                margin: 40px;
                color: #2c3e50;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #1a5276;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
            }}
            th, td {{
                border: 1px solid #bbb;
                padding: 8px 10px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {html_content}
    </body>
    </html>
    """

    temp_path = tempfile.mktemp(suffix=".pdf")
    HTML(string=html_full).write_pdf(temp_path)
    return temp_path
