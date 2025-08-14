# SteamNoodles – Automated Restaurant Feedback Agents (LangChain)
Jakshigan Jeyaseelan<br/>
NSBM Green University
--
Two LangChain agents for SteamNoodles:
1) **Feedback Response Agent** – classifies a single review (positive/negative/neutral) and generates a short, polite, context-aware reply.
2) **Sentiment Visualization Agent** – given a date range, plots daily counts of positive/negative/neutral reviews.

---

## ✨ Features
- LangChain + OpenAI GPT for sentiment + response
- Natural-language date ranges like *"last 7 days"* (via `dateparser`)
- Matplotlib plots saved to `outputs/`
- Sample dataset in `data/reviews.csv` (timestamp, text, sentiment)

---

## 🧰 Tech
- **Framework:** LangChain
- **LLM:** OpenAI (GPT via `langchain-openai`)
- **Data/Plots:** pandas, matplotlib
- **Dates:** dateparser
- **Env:** Python 3.10+

---

## 🚀 Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

> If you already have your dataset, replace `data/reviews.csv` with your file (must have columns: `timestamp`, `text`, `sentiment`). Timestamps should be ISO-like (e.g., `2025-08-10 14:32:00`).

---

## ▶️ Usage

### 1) Feedback Response Agent (single review)
```bash
python main.py respond --text "The noodles were fresh and tasty, but service was slow."
```
**Output:** prints predicted sentiment and the auto-reply.

### 2) Sentiment Visualization Agent (plot by date range)
Use natural language:
```bash
python main.py plot --range "last 7 days"
```
Or explicit dates (YYYY-MM-DD):
```bash
python main.py plot --start 2025-08-01 --end 2025-08-14
```
**Output:** saves a PNG to `outputs/` and prints the path.

---

## 🧪 Demo
Try:
```bash
python main.py respond --text "Loved the spicy miso ramen! Staff were friendly."
python main.py plot --range "last 7 days"
```
A demo plot will be generated from the included sample dataset.

---

## 🗂️ Project Structure
```
steamnoodles_feedback/
├── agents/
│   ├── feedback_response_agent.py
│   └── sentiment_plotting_agent.py
├── data/
│   └── reviews.csv
├── outputs/                         # plots appear here
├── utils/
│   └── llm_utils.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📌 Notes
- The plotting agent **does not** call the LLM; it relies on the `sentiment` column in the dataset for speed/cost control.
- The response agent uses the LLM to both classify and reply in one call to reduce latency and maintain consistency.
- For production, consider adding rate limits, retry logic, and a persistent vector store if you later add RAG-based personalization.

---

## ✅ Sample Prompts & Expected Outputs

**Respond**
- Prompt: `The noodles were fresh and delicious, but service was slow.`
- Output sentiment: `mixed` → normalized to `neutral`
- Auto-reply: `Thank you for sharing your feedback! We're thrilled you enjoyed our noodles and we're working on improving our service speed.`

**Plot**
- Prompt: `last 7 days`
- Output: A PNG showing daily counts for positive/negative/neutral from the sample dataset.

---

## 🧯 Troubleshooting
- If you see `ModuleNotFoundError: langchain_openai`, ensure you installed `langchain-openai` (dash!) and not `langchain_openai`.
- If plots are blank, confirm your date range overlaps with your dataset and that `timestamp` parses correctly.

---

## 📜 License
MIT – for this template. Verify any dataset license before redistribution.
