\
import argparse
from agents.feedback_response_agent import FeedbackResponseAgent
from agents.sentiment_plotting_agent import SentimentPlottingAgent

def run_respond(args):
    agent = FeedbackResponseAgent(model=args.model)
    res = agent.generate_reply(args.text)
    print(f"Sentiment: {res.sentiment}")
    print(f"Reply: {res.reply}")

def run_plot(args):
    agent = SentimentPlottingAgent()
    path = agent.plot_sentiment_trend(date_range=args.range, start=args.start, end=args.end, chart_type=args.chart)
    print(f"Saved plot to: {path}")

def build_parser():
    parser = argparse.ArgumentParser(description="SteamNoodles Feedback Agents (LangChain)")
    sub = parser.add_subparsers(dest="command", required=True)

    # Respond
    p_resp = sub.add_parser("respond", help="Classify sentiment & generate automated reply for a single review")
    p_resp.add_argument("--text", required=True, help="Customer review text")
    p_resp.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")
    p_resp.set_defaults(func=run_respond)

    # Plot
    p_plot = sub.add_parser("plot", help="Plot sentiment trend over a date range")
    p_plot.add_argument("--range", help='Natural language date range, e.g., "last 7 days" or "June 1 to June 15"')
    p_plot.add_argument("--start", help="Start date YYYY-MM-DD")
    p_plot.add_argument("--end", help="End date YYYY-MM-DD")
    p_plot.add_argument("--chart", choices=["line", "bar"], default="line", help="Chart type (default: line)")
    p_plot.set_defaults(func=run_plot)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
