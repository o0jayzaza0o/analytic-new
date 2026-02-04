# analytic-new
# 🌍 Global News Analysis with Llama 3.2 (Unsloth)

This project is an automated **Information Extraction & Sentiment Analysis pipeline** for international news. 
It utilizes a Small Language Model (SLM) to extract public figures and evaluate sentiment, specifically focusing on global leaders and tech figures.

## 🚀 Key Features
* **News Scraper:** Fetches real-time news from International RSS Feeds (Reuters/BBC).
* **AI Inference:** Powered by **Llama-3.2-3B-Instruct** optimized with **Unsloth** (4-bit quantization).
* **Focus Group Logic:** Specifically tracks figures like *Elon Musk, Donald Trump, Joe Biden*.
* **Interactive Dashboard:** Visualizes insights using **Streamlit**.

## 🛠 Tech Stack
* **Python 3.10+**
* **AI/ML:** `unsloth`, `torch`, `transformers`
* **Data Engineering:** `feedparser`, `beautifulsoup4`, `pandas`
* **Visualization:** `streamlit`, `plotly`

## 📊 How it works
1.  **Ingest:** Pull raw text from RSS feeds.
2.  **Process:** Feed text into Llama 3.2 with a custom engineered prompt for JSON extraction.
3.  **Clean:** Validate and sanitize JSON output (handle edge cases).
4.  **Visualize:** Display trends on Streamlit dashboard.

## 📂 Project Structure
```text
├── app.py                  # Streamlit Dashboard
├── global_news_analysis.csv # Sample Data
├── analysis_notebook.ipynb  # Colab Notebook (Source Code)
└── README.md
