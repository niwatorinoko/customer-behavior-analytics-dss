# Marketing Strategy Decision Support System

以下の2つのどちらかまたは両方を使い、マーケティング戦略の意思決定を支援するシステムです。

①顧客の購買履歴に基づいた顧客セグメンテーション

②売上履歴に基づいた商品販売集計・予測


## セットアップ
> pyenvの仮想環境でも実行できますが、今回はDockerを用いてシステムを起動する想定です。
### Docker Compose で動かす方法
Dockerfile があるため `docker-compose.yml` から起動できます。
```bash
docker compose build
docker compose up
```
ブラウザで `http://localhost:8501` にアクセスしてください。コードを編集するとコンテナ内にも反映されるよう、リポジトリ全体をコンテナにマウントしています。

### LLM用APIキーの設定
GeminiのAPIキーを `.env` に設定してから起動してください。
1. `.env.example` をコピーして `.env` を作成し、`GEMINI_API_KEY` を自身のキーに置き換える。
   ```bash
   cp .env.example .env
   ```
