\
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from dateparser import parse as parse_date

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reviews.csv")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

@dataclass
class SentimentPlotConfig:
    data_path: str = DEFAULT_DATA_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    chart_type: str = "line"   # "line" or "bar"


class SentimentPlottingAgent:
    """
    Agent 2: Given a date range, plots per-day counts by sentiment.
    Expects dataset with columns: timestamp, text, sentiment
    sentiment must be one of: positive, negative, neutral
    """

    def __init__(self, config: Optional[SentimentPlotConfig] = None):
        self.config = config or SentimentPlotConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path)
        # parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "sentiment"])
        # Normalize sentiment
        df["sentiment"] = df["sentiment"].str.lower().str.strip().map(
            lambda x: x if x in {"positive", "negative", "neutral"} else "neutral"
        )
        return df

    def _parse_range(self, date_range: Optional[str], start: Optional[str], end: Optional[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if date_range:
            # Natural language, e.g., "last 7 days"
            # We'll interpret "last N days" as (today - N + 1) .. today inclusive
            dr = date_range.lower().strip()
            if dr.startswith("last "):
                import re
                m = re.search(r"last\s+(\d+)\s+days?", dr)
                if m:
                    n = int(m.group(1))
                    end_dt = pd.Timestamp.today().normalize()
                    start_dt = end_dt - pd.Timedelta(days=n-1)
                    return start_dt, end_dt
            # Fallback to dateparser parsing like "June 1 to June 15"
            if "to" in dr:
                parts = [p.strip() for p in dr.split("to")]
                if len(parts) == 2:
                    s = parse_date(parts[0])
                    e = parse_date(parts[1])
                    if s and e:
                        return pd.Timestamp(s.date()), pd.Timestamp(e.date())
            # Single date fallback
            s = parse_date(dr)
            if s:
                s = pd.Timestamp(s.date())
                return s, s

        # Explicit start/end
        if start:
            s = pd.to_datetime(start).normalize()
        else:
            s = pd.Timestamp.min
        if end:
            e = pd.to_datetime(end).normalize()
        else:
            e = pd.Timestamp.max
        return s, e

    def plot_sentiment_trend(self, date_range: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None, chart_type: Optional[str] = None) -> str:
        df = self._load_data()
        s, e = self._parse_range(date_range, start, end)
        mask = (df["timestamp"].dt.normalize() >= s) & (df["timestamp"].dt.normalize() <= e)
        df = df.loc[mask].copy()
        if df.empty:
            raise ValueError("No data in the specified date range.")

        df["day"] = df["timestamp"].dt.date
        counts = df.groupby(["day", "sentiment"]).size().unstack(fill_value=0).sort_index()

        chart_type = (chart_type or self.config.chart_type).lower()
        plt.figure()
        if chart_type == "line":
            counts.plot(kind="line", marker="o")
        else:
            counts.plot(kind="bar")

        plt.title(f"SteamNoodles Sentiment by Day ({s.date()} to {e.date()})")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()

        fname = f"sentiment_trend_{s.date()}_to_{e.date()}.png".replace(":", "-")
        out_path = os.path.join(self.config.output_dir, fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
