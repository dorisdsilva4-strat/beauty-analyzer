"""
============================================================================
  DESI BEAUTY DATA ANALYZER  ·  v3 (News + Sentiment + YouTube)
  Real-time trend dashboard for an Indian beauty / makeup / skincare
  Instagram page.
----------------------------------------------------------------------------
  Sources : Google Trends (pytrends, GEO=IN)
            Google News RSS  + VADER sentiment      (no keys, always on)
            YouTube comments + VADER sentiment       (optional free API key)
  Output  : Pastel, Instagram-ready 1080x1080 charts to download & post
============================================================================

SETUP (local only; on Streamlit Cloud requirements.txt handles it)
    pip install streamlit pandas matplotlib pytrends requests feedparser vaderSentiment
    streamlit run desi_beauty_data_analyzer.py

YOUTUBE (optional — adds real audience comments):
  A free API key, ~5 min, works from the cloud:
    1. console.cloud.google.com  ->  create a project (any name)
    2. Search "YouTube Data API v3"  ->  Enable
    3. APIs & Services -> Credentials -> Create credentials -> API key
    4. Copy the key (starts with AIza...)
    5. Paste it into the app's sidebar -> "YouTube API key (optional)"
  No key = news + trends still work fully; YouTube tab just stays dormant.
============================================================================
"""

import io
import re
import time
import random
import html
import datetime as dt
from collections import Counter
from urllib.parse import quote_plus

import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

import streamlit as st

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except Exception:
    PYTRENDS_AVAILABLE = False
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except Exception:
    FEEDPARSER_AVAILABLE = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

KEYWORD_GROUPS = {
    "Lip Products":       ["kajal", "tinted lip balm", "lip tint", "matte lipstick"],
    "Skincare":           ["clear sunscreen", "kojic acid", "saffron serum"],
    "Cultural Catalysts": ["glass skin", "wedding makeup"],
}

NEWS_QUERIES = [
    "Indian skincare trends", "Indian makeup trends", "glass skin skincare India",
    "sunscreen India beauty", "lipstick launch India", "beauty brand India launch",
]

# Kept few on purpose: each YouTube search costs 100 quota units (of 10k/day).
YOUTUBE_QUERIES = [
    "Indian skincare routine review", "Indian makeup tutorial",
    "sunscreen India review", "glass skin India",
]

BRANDS = [
    "Nykaa", "Tira", "Sugar", "Minimalist", "L'Oreal", "Maybelline", "Lakme",
    "Plum", "Dot & Key", "The Ordinary", "Cetaphil", "Deconstruct", "Foxtale",
    "Renee", "Mamaearth", "Sunscoop", "Aqualogica", "Cosrx", "Innisfree",
]


# ---------------------------------------------------------------------------
# 2. AESTHETIC
# ---------------------------------------------------------------------------

PALETTE = {"bg": "#FFFDF9", "grid": "#EAE3DA", "ink": "#5A5150", "muted": "#A79B97"}
SERIES_COLORS = ["#E19AAE", "#A9C3A0", "#B7A6D6", "#F0B79A", "#9FC0D4",
                 "#CBA0C4", "#E7C98B", "#8FB8AE"]
POS, NEU, NEG = "#A9C3A0", "#B7A6D6", "#E19AAE"
SQUARE_IN, SQUARE_DPI = 10.8, 100


def _pick_font():
    preferred = ["Poppins", "Montserrat", "Nunito Sans", "Segoe UI",
                 "Helvetica Neue", "Arial"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return "DejaVu Sans"


def set_aesthetic_style():
    font = _pick_font()
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"], "font.family": font,
        "text.color": PALETTE["ink"], "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["ink"], "axes.titlecolor": PALETTE["ink"],
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "axes.linewidth": 0.8, "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6, "figure.dpi": SQUARE_DPI,
    })


def _new_square_fig():
    fig, ax = plt.subplots(figsize=(SQUARE_IN, SQUARE_IN), dpi=SQUARE_DPI)
    fig.subplots_adjust(left=0.14, right=0.94, top=0.84, bottom=0.16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _title(fig, ax, title, subtitle):
    fig.text(0.14, 0.93, title, fontsize=26, fontweight="bold",
             color=PALETTE["ink"], ha="left")
    fig.text(0.14, 0.885, subtitle, fontsize=13, color=PALETTE["muted"], ha="left")
    fig.text(0.94, 0.05, "@ your.beauty.page", fontsize=11,
             color=PALETTE["muted"], ha="right", style="italic")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SQUARE_DPI, facecolor=PALETTE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------

def _with_retry(fn, tries=3, base_delay=2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_delay * (2 ** i) + random.random())
    raise last


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 4. GOOGLE TRENDS
# ---------------------------------------------------------------------------

def _fallback_trends(groups, start, end):
    idx = pd.date_range(start=start, end=end, freq="D")
    data = {}
    for kws in groups.values():
        for kw in kws:
            rng = random.Random(sum(ord(c) for c in kw))
            level = rng.randint(25, 70)
            vals = []
            for _ in idx:
                level = max(3, min(100, level + rng.randint(-8, 9)))
                vals.append(level)
            data[kw] = vals
    df = pd.DataFrame(data, index=idx)
    df.index.name = "date"
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_trends_data(groups, geo, start, end):
    if not PYTRENDS_AVAILABLE:
        return _fallback_trends(groups, start, end), False
    timeframe = f"{start} {end}"
    frames = []
    try:
        pytrends = TrendReq(hl="en-IN", tz=330, timeout=(10, 25))
        for kws in groups.values():
            def _fetch(kws=kws):
                pytrends.build_payload(kws, cat=0, timeframe=timeframe, geo=geo)
                return pytrends.interest_over_time()
            df = _with_retry(_fetch)
            if df is None or df.empty:
                continue
            df = df.drop(columns=[c for c in ("isPartial",) if c in df.columns])
            frames.append(df)
            time.sleep(1.5)
        if not frames:
            return _fallback_trends(groups, start, end), False
        merged = pd.concat(frames, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged.index.name = "date"
        return merged, True
    except Exception:
        return _fallback_trends(groups, start, end), False


# ---------------------------------------------------------------------------
# 5. GOOGLE NEWS RSS
# ---------------------------------------------------------------------------

_FALLBACK_NEWS = [
    ("Nykaa reports strong demand for glass-skin serums this festive season", "Business Today"),
    ("Minimalist launches new SPF 50 clear sunscreen, sells out in days", "YourStory"),
    ("Consumers complain about greasy finish on some tinted lip balms", "The Print"),
    ("Tira expands offline stores as beauty retail booms in India", "Economic Times"),
    ("Sugar Cosmetics matte lipstick range wins praise for shade inclusivity", "Vogue India"),
    ("Dermatologists warn against overusing kojic acid for pigmentation", "Indian Express"),
    ("Wedding makeup trends 2026: dewy skin and bold kajal dominate", "Cosmopolitan India"),
    ("Aqualogica sunscreen faces backlash over white-cast claims", "The Quint"),
    ("Mamaearth parent Honasa posts steady quarter amid beauty slowdown", "Mint"),
    ("Foxtale vitamin C serum gains traction with Gen Z shoppers", "Hindustan Times"),
    ("Lakme rolls out saffron-infused serum for the Indian market", "Financial Express"),
    ("Deconstruct and Minimalist lead the budget skincare shake-up", "Forbes India"),
]


def _fallback_articles():
    now = dt.datetime.utcnow()
    return [{"title": t, "source": s,
             "published": (now - dt.timedelta(days=i)).strftime("%d %b %Y"),
             "link": "", "text": t, "query": "sample"}
            for i, (t, s) in enumerate(_FALLBACK_NEWS)]


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(queries, per_query=12):
    if not FEEDPARSER_AVAILABLE:
        return _fallback_articles(), False
    ua = "Mozilla/5.0 (desi-beauty-data-analyzer)"
    seen, articles = set(), []
    for q in queries:
        url = ("https://news.google.com/rss/search?q="
               + quote_plus(q) + "&hl=en-IN&gl=IN&ceid=IN:en")
        try:
            def _fetch(url=url):
                return feedparser.parse(url, agent=ua)
            feed = _with_retry(_fetch, tries=2, base_delay=2.0)
            for e in feed.entries[:per_query]:
                title = _clean(getattr(e, "title", ""))
                if not title or title in seen:
                    continue
                seen.add(title)
                source = ""
                if getattr(e, "source", None) and getattr(e.source, "title", None):
                    source = e.source.title
                articles.append({
                    "title": title, "source": source,
                    "published": getattr(e, "published", ""),
                    "link": getattr(e, "link", ""),
                    "text": title + ". " + _clean(getattr(e, "summary", "")),
                    "query": q})
            time.sleep(0.6)
        except Exception:
            continue
    return (articles, True) if articles else (_fallback_articles(), False)


# ---------------------------------------------------------------------------
# 6. YOUTUBE COMMENTS (optional, needs a free API key)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def get_youtube_comments(api_key, queries, videos_per_query=4, comments_per_video=25):
    """Return (comments, is_live). Empty + False when no/invalid key."""
    if not api_key:
        return [], False
    base = "https://www.googleapis.com/youtube/v3/"
    comments = []
    try:
        for q in queries:
            def _search(q=q):
                r = requests.get(base + "search", params={
                    "part": "snippet", "q": q, "type": "video",
                    "maxResults": videos_per_query, "regionCode": "IN",
                    "relevanceLanguage": "en", "order": "relevance",
                    "key": api_key}, timeout=20)
                r.raise_for_status()
                return r.json()
            data = _with_retry(_search, tries=2, base_delay=2.0)
            for item in data.get("items", []):
                vid = item.get("id", {}).get("videoId")
                vtitle = _clean(item.get("snippet", {}).get("title", ""))
                if not vid:
                    continue
                try:
                    rc = requests.get(base + "commentThreads", params={
                        "part": "snippet", "videoId": vid,
                        "maxResults": comments_per_video, "order": "relevance",
                        "textFormat": "plainText", "key": api_key}, timeout=20)
                    if rc.status_code != 200:      # e.g. comments disabled
                        continue
                    for th in rc.json().get("items", []):
                        sn = th["snippet"]["topLevelComment"]["snippet"]
                        txt = _clean(sn.get("textDisplay", ""))
                        if txt:
                            comments.append({"text": txt, "video": vtitle,
                                             "likes": sn.get("likeCount", 0),
                                             "query": q})
                except Exception:
                    continue
            time.sleep(0.3)
    except Exception:
        return comments, bool(comments)      # bad key / quota -> [] , False
    return comments, bool(comments)


# ---------------------------------------------------------------------------
# 7. SENTIMENT + AGGREGATION  (shared by news & youtube)
# ---------------------------------------------------------------------------

def score_items(items):
    if VADER_AVAILABLE:
        an = SentimentIntensityAnalyzer()
        for a in items:
            a["score"] = an.polarity_scores(a["text"])["compound"]
    else:
        pos = {"love", "best", "glow", "praise", "wins", "strong", "amazing", "great"}
        neg = {"warn", "backlash", "complain", "greasy", "worst", "avoid", "hate", "bad"}
        for a in items:
            t = a["text"].lower()
            a["score"] = 0.4 * sum(w in t for w in pos) - 0.4 * sum(w in t for w in neg)
    for a in items:
        s = a["score"]
        a["label"] = "positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral"
    return items


def brand_stats(items, brands, min_hits=1):
    rows = []
    for b in brands:
        bl = b.lower()
        hits = [a for a in items if bl in a["text"].lower()]
        if len(hits) >= min_hits:
            rows.append({"brand": b, "volume": len(hits),
                         "sentiment": sum(a["score"] for a in hits) / len(hits)})
    return sorted(rows, key=lambda r: r["volume"], reverse=True)


def overall_mix(items):
    c = Counter(a["label"] for a in items)
    return c.get("positive", 0), c.get("neutral", 0), c.get("negative", 0)


# ---------------------------------------------------------------------------
# 8. CHARTS
# ---------------------------------------------------------------------------

def chart_interest_over_time(df, group, kws):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    for i, kw in enumerate(kws):
        if kw in df.columns:
            ax.plot(df.index, df[kw], label=kw, linewidth=2.6,
                    color=SERIES_COLORS[i % len(SERIES_COLORS)], solid_capstyle="round")
    ax.set_ylabel("Search interest", fontsize=12)
    ax.set_ylim(0, 105)
    fig.autofmt_xdate(rotation=25)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=min(len(kws), 3), frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(PALETTE["ink"])
    _title(fig, ax, group, "Google Trends · India · search interest over time")
    return _png(fig)


def chart_trending_now(df, top_n=10):
    set_aesthetic_style()
    latest = df.iloc[-1].sort_values(ascending=True).tail(top_n)
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6); ax.grid(False, axis="y")
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(latest))]
    bars = ax.barh(latest.index, latest.values, color=colors, height=0.62)
    for bar, v in zip(bars, latest.values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{int(v)}", va="center", fontsize=11, color=PALETTE["muted"])
    ax.set_xlim(0, 108)
    ax.set_xlabel("Search interest (latest)", fontsize=12)
    _title(fig, ax, "Trending Right Now", "Google Trends · India · latest search interest")
    return _png(fig)


def chart_volume(rows, title, subtitle, xlabel, top_n=10):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6); ax.grid(False, axis="y")
    rows = rows[:top_n][::-1]
    if not rows:
        ax.text(0.5, 0.5, "No brand mentions yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color=PALETTE["muted"])
    else:
        labels = [r["brand"] for r in rows]
        vals = [r["volume"] for r in rows]
        colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(rows))]
        bars = ax.barh(labels, vals, color=colors, height=0.62)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.02,
                    bar.get_y() + bar.get_height() / 2, str(v),
                    va="center", fontsize=11, color=PALETTE["muted"])
        ax.margins(x=0.12)
    ax.set_xlabel(xlabel, fontsize=12)
    _title(fig, ax, title, subtitle)
    return _png(fig)


def chart_sentiment(rows, title, subtitle, top_n=10):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6); ax.grid(False, axis="y")
    rows = rows[:top_n][::-1]
    if not rows:
        ax.text(0.5, 0.5, "Not enough coverage to score", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color=PALETTE["muted"])
    else:
        labels = [r["brand"] for r in rows]
        vals = [r["sentiment"] for r in rows]
        colors = [POS if v >= 0.05 else NEG if v <= -0.05 else NEU for v in vals]
        ax.barh(labels, vals, color=colors, height=0.62)
        ax.axvline(0, color=PALETTE["grid"], linewidth=1.2)
        ax.set_xlim(-1, 1)
    ax.set_xlabel("Average sentiment  (left = negative · right = positive)", fontsize=12)
    _title(fig, ax, title, subtitle)
    return _png(fig)


# ---------------------------------------------------------------------------
# 9. STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------

def _dl(label, png, name):
    st.download_button(label, data=png, file_name=name, mime="image/png",
                       use_container_width=True)


def _sentiment_tab(items, source_label, vol_xlabel, key_prefix):
    pos, neu, neg = overall_mix(items)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Items", len(items)); m2.metric("Positive", pos)
    m3.metric("Neutral", neu); m4.metric("Negative", neg)
    rows = brand_stats(items, BRANDS)
    l, r = st.columns(2)
    with l:
        p = chart_volume(rows, "Who's Being Talked About",
                         f"{source_label} · brand share of chatter", vol_xlabel)
        st.image(p, use_container_width=True)
        _dl("⬇️ Download (1080×1080)", p, f"{key_prefix}_volume.png")
    with r:
        p = chart_sentiment(rows, "How People Feel",
                            f"{source_label} · avg sentiment per brand")
        st.image(p, use_container_width=True)
        _dl("⬇️ Download (1080×1080)", p, f"{key_prefix}_sentiment.png")
    return rows


def main():
    st.set_page_config(page_title="Desi Beauty Data Analyzer",
                       page_icon="💄", layout="wide")
    st.markdown("""<style>
        .stApp{background-color:#FFFDF9;}
        h1,h2,h3,h4{color:#5A5150;}
        section[data-testid="stSidebar"]{background-color:#FCE9EC;}
        </style>""", unsafe_allow_html=True)

    st.title("💄 Desi Beauty Data Analyzer")
    st.caption("Real-time Indian beauty · makeup · skincare trends → Instagram-ready charts")

    with st.sidebar:
        st.header("Controls")
        today = dt.date.today()
        default_start = today - dt.timedelta(days=90)
        dr = st.date_input("Trends date range", value=(default_start, today),
                           max_value=today)
        start_date, end_date = dr if isinstance(dr, tuple) and len(dr) == 2 \
            else (default_start, today)
        with st.expander("YouTube API key (optional)"):
            st.caption("Adds real audience comments. Free key, ~5 min: "
                       "console.cloud.google.com → enable YouTube Data API v3 → "
                       "create an API key → paste below.")
            yt_key = st.text_input("YouTube Data API v3 key", type="password")
        if st.button("🔄 Refresh real-time data", use_container_width=True):
            st.cache_data.clear()
        if not FEEDPARSER_AVAILABLE:
            st.warning("Add `feedparser` to requirements.txt for live news.")
        if not VADER_AVAILABLE:
            st.warning("Add `vaderSentiment` to requirements.txt for accurate sentiment.")

    start_s, end_s = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    with st.spinner("Pulling Google Trends (India)…"):
        trends_df, trends_live = get_trends_data(KEYWORD_GROUPS, "IN", start_s, end_s)
    with st.spinner("Pulling news + scoring sentiment…"):
        articles = score_items(get_news(NEWS_QUERIES)[0])
        news_live = articles and articles[0].get("query") != "sample"
    yt_comments, yt_live = [], False
    if yt_key:
        with st.spinner("Pulling YouTube comments…"):
            yt_comments, yt_live = get_youtube_comments(yt_key, YOUTUBE_QUERIES)
            yt_comments = score_items(yt_comments)

    c1, c2, c3 = st.columns(3)
    with c1:
        (st.success if trends_live else st.info)(
            "Trends: LIVE ✅" if trends_live else "Trends: SAMPLE ⚠️")
    with c2:
        (st.success if news_live else st.info)(
            f"News: LIVE ✅ ({len(articles)})" if news_live else "News: SAMPLE ⚠️")
    with c3:
        if not yt_key:
            st.info("YouTube: add key to enable")
        else:
            (st.success if yt_live else st.warning)(
                f"YouTube: LIVE ✅ ({len(yt_comments)})" if yt_live
                else "YouTube: key/quota issue ⚠️")

    st.divider()
    t_trends, t_news, t_yt, t_data = st.tabs(
        ["📈 Google Trends", "📰 News & Sentiment", "▶️ YouTube", "🧾 Raw data"])

    with t_trends:
        group = st.selectbox("Keyword group", list(KEYWORD_GROUPS.keys()))
        kws = KEYWORD_GROUPS[group]
        l, r = st.columns(2)
        with l:
            p = chart_interest_over_time(trends_df, group, kws)
            st.image(p, use_container_width=True)
            _dl("⬇️ Download (1080×1080)", p,
                f"trends_{group.lower().replace(' ', '_')}.png")
        with r:
            p = chart_trending_now(trends_df)
            st.image(p, use_container_width=True)
            _dl("⬇️ Download 'Trending Now' (1080×1080)", p, "trending_now.png")

    with t_news:
        _sentiment_tab(articles, "Google News · India", "Articles mentioning brand", "news")
        st.subheader("Headlines driving the numbers")
        emoji = {"positive": "🟢", "neutral": "🟣", "negative": "🔴"}
        for a in sorted(articles, key=lambda x: x["score"], reverse=True):
            src = f" · {a['source']}" if a["source"] else ""
            line = f"{emoji[a['label']]} **{a['title']}**{src}"
            if a["link"]:
                line += f"  [↗]({a['link']})"
            st.markdown(line)

    with t_yt:
        if not yt_key:
            st.info("**YouTube is optional and currently off.** Add a free API key "
                    "in the sidebar to pull real audience comments and score their "
                    "sentiment. It works from the cloud and takes about 5 minutes to set up.")
        elif not yt_comments:
            st.warning("No comments came back. Usually a wrong/restricted key or the "
                       "daily quota is used up (resets at midnight Pacific). "
                       "Double-check the key in the sidebar.")
        else:
            _sentiment_tab(yt_comments, "YouTube · India", "Comments mentioning brand", "youtube")
            st.subheader("Top comments driving the mood")
            emoji = {"positive": "🟢", "neutral": "🟣", "negative": "🔴"}
            for a in sorted(yt_comments, key=lambda x: x["likes"], reverse=True)[:20]:
                st.markdown(f"{emoji[a['label']]} {a['text'][:200]}  "
                            f"·  👍{a['likes']}")

    with t_data:
        st.subheader("Google Trends — interest over time")
        st.dataframe(trends_df, use_container_width=True)
        st.download_button("⬇️ Trends CSV", data=trends_df.to_csv().encode("utf-8"),
                           file_name="google_trends_india.csv", mime="text/csv")
        st.subheader("News + sentiment")
        ndf = pd.DataFrame([{"title": a["title"], "source": a["source"],
                             "sentiment": round(a["score"], 3), "label": a["label"],
                             "link": a["link"]} for a in articles])
        st.dataframe(ndf, use_container_width=True)
        if yt_comments:
            st.subheader("YouTube comments + sentiment")
            ydf = pd.DataFrame([{"comment": a["text"], "video": a["video"],
                                 "likes": a["likes"], "sentiment": round(a["score"], 3),
                                 "label": a["label"]} for a in yt_comments])
            st.dataframe(ydf, use_container_width=True)


if __name__ == "__main__":
    main()
