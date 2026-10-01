"""
app.py — Movie Recommendation System (Attractive Streamlit UI)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from urllib.parse import quote
from recommender import MovieRecommender

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.2s;
    display: block;
    margin: 4px 0;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(139, 92, 246, 0.2);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94a3b8 !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.6) !important;
}

/* ── Selectbox / Slider ── */
[data-testid="stSelectbox"] > div,
.stSlider {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}

/* ── Success / Info / Warning ── */
.stAlert { border-radius: 12px !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Headings ── */
h1 { background: linear-gradient(135deg, #a78bfa, #60a5fa);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 700 !important; }
h2, h3 { color: #e2e8f0 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #a78bfa !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Genre → Color map ─────────────────────────────────────────────────────────
GENRE_COLORS = {
    "Drama":      ("#6366f1", "#4338ca"),
    "Sci-Fi":     ("#06b6d4", "#0284c7"),
    "Romance":    ("#ec4899", "#db2777"),
    "Thriller":   ("#f59e0b", "#d97706"),
    "Comedy":     ("#10b981", "#059669"),
    "Adventure":  ("#8b5cf6", "#7c3aed"),
    "Horror":     ("#ef4444", "#dc2626"),
    "Mystery":    ("#64748b", "#475569"),
    "Historical": ("#a16207", "#854d0e"),
    "Documentary":("#14b8a6", "#0f766e"),
    "Action":     ("#f97316", "#ea580c"),
    "Crime":      ("#6b21a8", "#581c87"),
    "Family":     ("#84cc16", "#65a30d"),
}

def genre_color(genre):
    return GENRE_COLORS.get(genre, ("#7c3aed", "#4f46e5"))


def movie_poster_url(title, genre="Drama"):
    bg, _ = genre_color(genre)
    title_text = quote((title or "Movie").replace("&", "and"), safe="")
    return f"https://placehold.co/600x900/{bg.lstrip('#')}/F8FAFC?text={title_text}"


def apply_genre_filter(results, genre_filter, user_id=None):
    if genre_filter == "All Genres" or results.empty or "Genre" not in results.columns:
        return results

    exact = results[results["Genre"] == genre_filter]
    if not exact.empty:
        return exact

    if user_id is not None:
        history = rec.get_user_history(user_id)
        if not history.empty:
            preferred = history["Genre"].dropna().value_counts().head(3).index.tolist()
            if preferred:
                ranked = results.copy()
                ranked["_match_bonus"] = ranked["Genre"].apply(
                    lambda g: 2 if g in preferred else 1 if g == genre_filter else 0
                )
                ranked = ranked.sort_values("_match_bonus", ascending=False)
                return ranked.drop(columns=["_match_bonus"]).head(len(ranked))

    return results.head(len(results))

# ── Load Model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_recommender():
    return MovieRecommender()

rec = load_recommender()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:3rem;'>🎬</div>
        <div style='font-size:1.4rem; font-weight:700; color:#a78bfa;'>CineAI</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Personalized Recommendations</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Home", "🎯  Recommendations", "📊  Explore Data"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.75rem; text-align:center; padding:10px;'>
        200 Users · 100 Movies<br>
        5 ML Algorithms<br>
        <span style='color:#7c3aed;'>Powered by Collaborative Filtering</span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # Hero
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px 0;'>
        <h1 style='font-size:3.2rem; margin-bottom:8px;'>🎬 CineAI</h1>
        <p style='color:#94a3b8; font-size:1.15rem; max-width:600px; margin:0 auto;'>
            Your AI-powered personal movie guide — discover films you'll love
            using advanced machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎥 Movies", len(rec.movies))
    c2.metric("👥 Users", len(rec.matrix))
    c3.metric("⭐ Total Ratings", f"{int((rec.matrix > 0).sum().sum()):,}")
    c4.metric("🎭 Genres", rec.movies["Genre"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    # Algorithm cards
    st.markdown("### 🧠 How CineAI Works")
    st.markdown("<br>", unsafe_allow_html=True)

    algo_cols = st.columns(5)
    algos = [
        ("👤", "User-CF", "Finds users with similar tastes and recommends what they loved"),
        ("🎬", "Item-CF", "Finds movies similar to ones you already enjoyed"),
        ("📝", "Content", "Matches genre, director & year to your taste profile"),
        ("🔢", "SVD", "Matrix Factorization to uncover hidden viewing patterns"),
        ("🔀", "Hybrid", "Combines multiple algorithms for best accuracy"),
    ]
    for col, (icon, name, desc) in zip(algo_cols, algos):
        col.markdown(f"""
<div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
            border-radius:16px; padding:20px 14px; text-align:center; height:180px;'>
    <div style='font-size:2rem; margin-bottom:10px;'>{icon}</div>
    <div style='font-weight:600; color:#a78bfa; font-size:0.95rem; margin-bottom:8px;'>{name}</div>
    <div style='color:#64748b; font-size:0.78rem; line-height:1.5;'>{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Top Rated Movies
    st.markdown("### 🏆 Top Rated Movies")
    top = rec.get_top_movies(8)
    if not top.empty:
        cols = st.columns(4)
        for i, (_, row) in enumerate(top.iterrows()):
            genre = row.get("Genre", "Drama")
            g1, g2 = genre_color(genre)
            title = row.get("Title", "Unknown")
            year  = row.get("Year", "")
            director = row.get("Director", "")
            avg  = row.get("AvgRating", 0)
            num  = int(row.get("NumRatings", 0))
            stars = "⭐" * int(round(avg))
            poster = movie_poster_url(title, genre)
            with cols[i % 4]:
                st.image(poster, use_container_width=True, output_format="JPEG")
                st.markdown(f"""
<div style='background:linear-gradient(135deg,{g1}22,{g2}11);
            border:1px solid {g1}44; border-radius:16px;
            padding:18px; margin-top:12px; margin-bottom:14px;'>
    <div style='background:linear-gradient(135deg,{g1},{g2});
                border-radius:8px; display:inline-block;
                padding:3px 10px; font-size:0.7rem; color:white;
                font-weight:600; margin-bottom:10px;'>{genre}</div>
    <div style='font-weight:700; color:#e2e8f0; font-size:1rem;
                margin-bottom:4px; line-height:1.3;'>{title}</div>
    <div style='color:#64748b; font-size:0.78rem; margin-bottom:8px;'>
        📅 {year} &nbsp;|&nbsp; 🎬 {director}</div>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <span style='color:#fbbf24; font-size:0.85rem;'>{stars}</span>
        <span style='color:#a78bfa; font-weight:600; font-size:0.9rem;'>{avg:.1f} / 5</span>
    </div>
    <div style='color:#475569; font-size:0.72rem; margin-top:4px;'>{num} ratings</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "🎯  Recommendations":

    st.markdown("""
    <div style='padding: 30px 0 10px 0;'>
        <h1 style='font-size:2.4rem; margin-bottom:6px;'>🎯 Get Recommendations</h1>
        <p style='color:#64748b;'>Tell us who you are and we'll find the perfect movies for you.</p>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        user_id = st.selectbox("👤 User ID", rec.matrix.index.tolist())
    with col2:
        algorithm = st.selectbox("🧠 Algorithm", ["Hybrid", "User-CF", "Item-CF", "Content-Based", "SVD"])
    with col3:
        genres = ["All Genres"] + sorted(rec.movies["Genre"].dropna().unique().tolist())
        genre_filter = st.selectbox("🎭 Genre", genres)
    with col4:
        n_recs = st.slider("Top N", 3, 10, 6)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn, _ = st.columns([2, 6])
    with col_btn:
        go = st.button("✨ Find My Movies", use_container_width=True)

    if go:
        with st.spinner("🎬 Finding the best movies for you..."):
            candidate_n = max(n_recs * 10, 20)
            if algorithm == "User-CF":
                results = rec.user_cf(user_id, n=candidate_n)
            elif algorithm == "Item-CF":
                results = rec.item_cf(user_id, n=candidate_n)
            elif algorithm == "Content-Based":
                results = rec.content_based(user_id, n=candidate_n)
            elif algorithm == "SVD":
                results = rec.svd_recommend(user_id, n=candidate_n)
            else:
                results = rec.hybrid(user_id, n=candidate_n)

            results = apply_genre_filter(results, genre_filter, user_id=user_id)
            results = results.head(n_recs)

        if results.empty:
            st.warning("⚠️ No recommendations found. Try a different algorithm or remove the genre filter.")
        else:
            st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(79,70,229,0.1));
            border:1px solid rgba(124,58,237,0.3); border-radius:14px;
            padding:14px 20px; margin: 10px 0 24px 0;'>
    <span style='color:#a78bfa; font-weight:600;'>✅ {len(results)} movies found</span>
    <span style='color:#64748b;'> for <b style="color:#e2e8f0">{user_id}</b>
    using <b style="color:#e2e8f0">{algorithm}</b></span>
</div>
""", unsafe_allow_html=True)

            # Movie cards — 3 per row
            num_rows = (len(results) + 2) // 3
            idx = 0
            for _ in range(num_rows):
                row_cols = st.columns(3)
                for col in row_cols:
                    if idx >= len(results):
                        break
                    row = results.iloc[idx]
                    genre   = row.get("Genre", "Drama")
                    g1, g2  = genre_color(genre)
                    title   = row.get("Title", row["MovieID"])
                    year    = row.get("Year", "N/A")
                    director = row.get("Director", "N/A")
                    duration = row.get("Duration (min)", "N/A")
                    confidence = row.get("Confidence", 0)
                    rank = idx + 1
                    poster = movie_poster_url(title, genre)

                    # Confidence bar width
                    bar_w = int(confidence) if confidence else 0

                    with col:
                        st.image(poster, use_container_width=True, output_format="JPEG")
                        st.markdown(f"""
<div style='background:linear-gradient(145deg,rgba(15,12,41,0.9),rgba(48,43,99,0.6));
            border:1px solid {g1}55; border-radius:20px;
            padding:18px; margin-top:12px; margin-bottom:16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
        <div style='background:rgba(255,255,255,0.07); border-radius:8px;
                    width:32px; height:32px; display:flex; align-items:center;
                    justify-content:center; font-weight:700; color:#94a3b8; font-size:0.85rem;'>
            #{rank}
        </div>
        <div style='background:linear-gradient(135deg,{g1},{g2});
                    border-radius:20px; padding:4px 12px;
                    font-size:0.72rem; color:white; font-weight:600;'>
            {genre}
        </div>
    </div>
    <div style='font-size:1.08rem; font-weight:700; color:#f1f5f9;
                margin-bottom:10px; line-height:1.35;'>{title}</div>
    <div style='color:#64748b; font-size:0.8rem; line-height:1.7;'>
        📅 <span style='color:#94a3b8;'>{year}</span> &nbsp;|&nbsp;
        🎬 <span style='color:#94a3b8;'>{director}</span><br>
        ⏱️ <span style='color:#94a3b8;'>{duration} min</span>
    </div>
    <div style='display:flex; justify-content:space-between; align-items:center; margin-top:12px;'>
        <span style='color:#475569; font-size:0.72rem;'>Match</span>
        <span style='color:{g1}; font-weight:600; font-size:0.8rem;'>{bar_w}%</span>
    </div>
    <div style='background:rgba(255,255,255,0.07); border-radius:6px; height:6px; margin-top:4px;'>
        <div style='background:linear-gradient(90deg,{g1},{g2}); width:{bar_w}%; height:6px; border-radius:6px;'></div>
    </div>
</div>
""", unsafe_allow_html=True)
                    idx += 1

            # Tabs for extra info
            st.markdown("<br>", unsafe_allow_html=True)
            t1, t2 = st.tabs(["🔍 Why These Movies?", "📜 Your Watch History"])

            with t1:
                if algorithm in ["User-CF", "Hybrid"]:
                    similar_users = rec.get_similar_users(user_id, n=5)
                    if not similar_users.empty:
                        st.markdown("**👥 Your Top Similar Users**")
                        for _, su_row in similar_users.iterrows():
                            sim_pct = int(su_row["Similarity"] * 100)
                            st.markdown(f"""
<div style='display:flex; align-items:center; gap:12px; padding:10px 0;
            border-bottom:1px solid rgba(255,255,255,0.05);'>
    <div style='background:linear-gradient(135deg,#7c3aed,#4f46e5);
                border-radius:50%; width:36px; height:36px;
                display:flex; align-items:center; justify-content:center;
                font-size:0.75rem; font-weight:700; color:white;'>
        {su_row['UserID'][1:]}
    </div>
    <div style='flex:1;'>
        <div style='color:#e2e8f0; font-weight:500; font-size:0.85rem;'>
            {su_row['UserID']}</div>
        <div style='background:rgba(255,255,255,0.06); border-radius:4px;
                    height:4px; margin-top:6px;'>
            <div style='background:linear-gradient(90deg,#7c3aed,#4f46e5);
                        width:{sim_pct}%; height:4px; border-radius:4px;'></div>
        </div>
    </div>
    <div style='color:#a78bfa; font-weight:600; font-size:0.85rem;'>
        {su_row['Similarity']:.2%}
    </div>
</div>
""", unsafe_allow_html=True)

                show_cols = [c for c in ["MovieID","Title","Genre","Year","Director","score","Confidence"]
                             if c in results.columns]
                st.markdown("<br>**📋 Recommendation Details**")
                st.dataframe(results[show_cols].reset_index(drop=True), use_container_width=True)

            with t2:
                history = rec.get_user_history(user_id)
                if not history.empty:
                    show_cols = [c for c in ["Title","Genre","Year","Director","Rating"]
                                 if c in history.columns]
                    st.dataframe(history[show_cols].reset_index(drop=True), use_container_width=True)
                else:
                    st.info("No watch history found for this user.")

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — EXPLORE DATA
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  Explore Data":

    st.markdown("""
    <div style='padding: 30px 0 10px 0;'>
        <h1 style='font-size:2.4rem; margin-bottom:6px;'>📊 Explore the Data</h1>
        <p style='color:#64748b;'>Dive into genre trends, rating patterns, and user activity.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎭 Genres", "⭐ Ratings", "👤 Users"])

    # Matplotlib dark theme
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor":   "none",
        "axes.edgecolor":   "#334155",
        "axes.labelcolor":  "#94a3b8",
        "xtick.color":      "#64748b",
        "ytick.color":      "#64748b",
        "text.color":       "#e2e8f0",
        "grid.color":       "#1e293b",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })

    # ── Tab 1: Genres ──
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        genre_stats = rec.get_genre_stats()

        if not genre_stats.empty:
            genre_stats.columns = ["Genre", "Count"]

            # Summary pills
            pill_cols = st.columns(min(len(genre_stats), 6))
            for i, (_, gr) in enumerate(genre_stats.head(6).iterrows()):
                g1, g2 = genre_color(gr["Genre"])
                pill_cols[i].markdown(f"""
<div style='background:linear-gradient(135deg,{g1}22,{g2}11);
            border:1px solid {g1}55; border-radius:12px;
            padding:14px; text-align:center;'>
    <div style='font-weight:700; color:{g1}; font-size:1.4rem;'>{gr["Count"]}</div>
    <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>{gr["Genre"]}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_gs = genre_stats.sort_values("Count")
            colors = [genre_color(g)[0] for g in sorted_gs["Genre"]]
            bars = ax.barh(sorted_gs["Genre"], sorted_gs["Count"],
                           color=colors, height=0.65, edgecolor="none")
            for bar, count in zip(bars, sorted_gs["Count"]):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), va="center", ha="left",
                        color="#94a3b8", fontsize=10, fontweight="600")
            ax.set_xlabel("Number of Movies", fontsize=10)
            ax.set_title("Movies by Genre", fontsize=14, fontweight="bold", pad=15, color="#e2e8f0")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(0, sorted_gs["Count"].max() + 2)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── Tab 2: Ratings ──
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        rating_dist = rec.get_rating_distribution()

        if not rating_dist.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Rating",   f"{rating_dist.mean():.2f} ⭐")
            c2.metric("Total Ratings", f"{len(rating_dist):,}")
            c3.metric("5-Star Ratings", f"{(rating_dist==5).sum():,}")
            c4.metric("1-Star Ratings", f"{(rating_dist==1).sum():,}")

            st.markdown("<br>", unsafe_allow_html=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Bar chart per rating
            rating_counts = rating_dist.value_counts().sort_index()
            bar_colors = ["#ef4444","#f97316","#eab308","#22c55e","#7c3aed"]
            axes[0].bar(rating_counts.index, rating_counts.values,
                        color=bar_colors[:len(rating_counts)],
                        width=0.6, edgecolor="none")
            axes[0].set_xticks([1,2,3,4,5])
            axes[0].set_xticklabels(["1★","2★","3★","4★","5★"])
            axes[0].set_title("Rating Distribution", fontsize=13, fontweight="bold")
            axes[0].set_ylabel("Number of Ratings")
            axes[0].spines["top"].set_visible(False)
            axes[0].spines["right"].set_visible(False)
            axes[0].yaxis.grid(True)

            # Pie chart
            labels = [f"{int(r)}★" for r in rating_counts.index]
            axes[1].pie(rating_counts.values, labels=labels,
                        colors=bar_colors[:len(rating_counts)],
                        autopct="%1.1f%%", startangle=140,
                        textprops={"color":"#94a3b8", "fontsize":10},
                        wedgeprops={"edgecolor":"none","linewidth":2})
            axes[1].set_title("Rating Share", fontsize=13, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── Tab 3: Users ──
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        ratings_per_user = (rec.matrix > 0).sum(axis=1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Movies / User", f"{ratings_per_user.mean():.1f}")
        c2.metric("Most Active User",  f"{ratings_per_user.max()} movies")
        c3.metric("Least Active User", f"{ratings_per_user.min()} movies")

        st.markdown("<br>", unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of ratings per user
        axes[0].hist(ratings_per_user, bins=20,
                     color="#7c3aed", edgecolor="none", alpha=0.85)
        axes[0].set_title("Ratings per User", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Movies Rated")
        axes[0].set_ylabel("Number of Users")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].yaxis.grid(True)

        # Top 10 most active users
        top_users = ratings_per_user.sort_values(ascending=False).head(10)
        colors_u = plt.cm.Purples(np.linspace(0.5, 0.9, len(top_users)))[::-1]
        axes[1].barh(top_users.index[::-1], top_users.values[::-1],
                     color=colors_u, edgecolor="none")
        axes[1].set_title("Top 10 Most Active Users", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Movies Rated")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].xaxis.grid(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔥 Most Rated Movies")
        most_rated = (rec.matrix > 0).sum().sort_values(ascending=False).head(10)
        most_rated_df = most_rated.reset_index()
        most_rated_df.columns = ["MovieID", "NumRatings"]
        most_rated_df = most_rated_df.merge(
            rec.movies[["MovieID","Title","Genre"]], on="MovieID", how="left")

        cols = st.columns(5)
        for i, (_, mrow) in enumerate(most_rated_df.head(10).iterrows()):
            g1, g2 = genre_color(mrow.get("Genre","Drama"))
            cols[i % 5].markdown(f"""
<div style='background:linear-gradient(135deg,{g1}18,{g2}0a);
            border:1px solid {g1}44; border-radius:14px;
            padding:14px; margin-bottom:12px; text-align:center;'>
    <div style='background:linear-gradient(135deg,{g1},{g2});
                border-radius:6px; padding:2px 8px; font-size:0.68rem;
                color:white; font-weight:600; display:inline-block;
                margin-bottom:8px;'>{mrow.get("Genre","")}</div>
    <div style='color:#e2e8f0; font-weight:600; font-size:0.85rem;
                line-height:1.3; margin-bottom:8px;'>{mrow.get("Title","")}</div>
    <div style='color:#a78bfa; font-weight:700; font-size:1.1rem;'>{int(mrow["NumRatings"])}</div>
    <div style='color:#475569; font-size:0.68rem;'>ratings</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:40px 0 20px 0;
            color:#334155; font-size:0.78rem; border-top:1px solid rgba(255,255,255,0.05);
            margin-top:40px;'>
    🎬 <b style='color:#475569'>CineAI</b> · AI/ML Movie Recommendation System ·
    Built with Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
