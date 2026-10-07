"""AI Data Analyst — Generative AI for automating data analysis tasks.

Run:  streamlit run app.py
"""
import io
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
APP_DIR = Path(__file__).parent
CACHE_FILE = APP_DIR / "demo_cache.json"

PROVIDERS = {
    "Groq (Llama)": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.5-flash",
    },
}

# Rough minutes a human analyst would spend on each task in Excel — used for the "time saved" counter.
MANUAL_MINUTES = {"profile": 15, "clean": 5, "query": 8, "insights": 25}

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- state
def init_state():
    defaults = {
        "df": None, "df_original": None, "file_name": None,
        "history": [], "applied_fixes": [], "fix_suggestions": None,
        "insights": None, "minutes_saved": 0, "seconds_spent": 0.0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


# ---------------------------------------------------------------- cache (offline safety net)
def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def norm(q):
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


# ---------------------------------------------------------------- LLM
def get_client():
    p = PROVIDERS[st.session_state.provider]
    key = st.session_state.get("api_key") or os.getenv(p["key_env"])
    if not key:
        return None, None
    from openai import OpenAI
    model = os.getenv(p["model_env"], p["default_model"])
    return OpenAI(api_key=key, base_url=p["base_url"]), model


def llm(messages, temperature=0.1):
    client, model = get_client()
    if client is None:
        raise RuntimeError("No API key. Add one in the sidebar or switch on Offline demo mode.")
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


# ---------------------------------------------------------------- data context for the LLM
def data_context(df):
    buf = io.StringIO()
    buf.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\nColumns and dtypes:\n")
    for c in df.columns:
        sample = df[c].dropna().unique()[:5]
        buf.write(f"- {c} ({df[c].dtype}), nulls={df[c].isna().sum()}, examples={list(sample)}\n")
    buf.write("\nFirst 5 rows:\n" + df.head().to_string())
    return buf.getvalue()


CODE_SYSTEM = """You are an expert data analyst who writes pandas code.
A DataFrame named `df` is already loaded. These are already available: pd, np, px (plotly.express), go (plotly.graph_objects).
Rules:
- Do NOT import anything. Do NOT read or write files. Do NOT modify df in place; work on copies.
- Put the final answer in a variable named `result` (a DataFrame, Series, number or string).
- If a chart helps, create a plotly figure named `fig` with a clear title and axis labels.
- Set `summary` to one plain-English sentence stating the answer with the key number(s), using f-strings computed from the data.
- If a date column is stored as text, convert it inside your code with pd.to_datetime(..., dayfirst=True, errors='coerce').
Return ONLY one ```python code block."""


# ---------------------------------------------------------------- safe execution
BLOCKED = ["import ", "open(", "__", "exec(", "eval(", "os.", "sys.", "subprocess", "shutil",
           "to_csv", "to_excel", "to_pickle", "globals", "locals", "getattr", "setattr", "compile("]

SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ["len", "range", "min", "max", "sum", "abs", "round", "sorted", "list", "dict",
                           "set", "tuple", "str", "int", "float", "bool", "enumerate", "zip", "map",
                           "filter", "any", "all", "isinstance", "print", "reversed", "__import__"]}
# __import__ stays available because pandas/plotly lazily import their own submodules at runtime;
# generated code still cannot import anything, since the text filter above rejects "import " and "__".


def run_code(code, df):
    for b in BLOCKED:
        if b in code:
            raise ValueError(f"Blocked for safety: generated code contains `{b.strip()}`")
    ns = {"df": df.copy(), "pd": pd, "np": np, "px": px, "go": go}
    exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
    if "result" not in ns and "fig" not in ns:
        raise ValueError("Code ran but did not set `result` or `fig`.")
    return ns.get("result"), ns.get("fig"), ns.get("summary")


def ask_question(question, df):
    """Question -> code -> run. Retries once by sending the error back to the model (self-correction)."""
    cache = load_cache()
    key = norm(question)
    attempts = []

    if st.session_state.offline or (st.session_state.use_cache and key in cache):
        if key not in cache:
            raise RuntimeError("Offline mode: this question isn't in the demo cache. "
                               "Ask one of the cached questions from the sidebar, or switch offline mode off.")
        code = cache[key]["code"]
        result, fig, summary = run_code(code, df)
        return {"code": code, "result": result, "fig": fig, "summary": summary,
                "source": "cache", "attempts": attempts}

    messages = [
        {"role": "system", "content": CODE_SYSTEM},
        {"role": "user", "content": f"Dataset:\n{data_context(df)}\n\nQuestion: {question}"},
    ]
    for attempt in range(2):
        reply = llm(messages)
        code = extract_code(reply)
        try:
            result, fig, summary = run_code(code, df)
            cache[key] = {"question": question, "code": code}
            save_cache(cache)
            return {"code": code, "result": result, "fig": fig, "summary": summary,
                    "source": "live", "attempts": attempts}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            attempts.append({"code": code, "error": err})
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"That code failed with:\n{err}\nFix it and return the full corrected code."}]
    raise RuntimeError(f"The AI could not produce working code after 2 attempts. Last error: {attempts[-1]['error']}")


# ---------------------------------------------------------------- profiling + cleaning
def profile(df):
    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
        "example": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else None for c in df.columns],
    })
    return prof


def rule_based_fixes(df):
    """Deterministic issue detection. Used offline, and as a fallback if the LLM reply can't be parsed."""
    fixes = []
    dups = int(df.duplicated().sum())
    if dups:
        fixes.append({"issue": f"{dups} exact duplicate rows", "fix": "Drop duplicate rows",
                      "code": "df = df.drop_duplicates().reset_index(drop=True)"})
    for c in df.select_dtypes(include=["object", "string"]):
        s = df[c].dropna().astype(str)
        if s.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$").mean() > 0.9:
            fixes.append({"issue": f"`{c}` is stored as text, not a date", "fix": f"Convert `{c}` to datetime (day-first)",
                          "code": f"df['{c}'] = pd.to_datetime(df['{c}'], dayfirst=True, errors='coerce')"})
            continue
        canon = s.str.strip().str.title()
        if canon.nunique() < s.nunique():
            fixes.append({"issue": f"`{c}` has inconsistent labels ({s.nunique()} variants for {canon.nunique()} values)",
                          "fix": f"Standardise `{c}` (trim spaces, title case)",
                          "code": f"df['{c}'] = df['{c}'].str.strip().str.title()"})
    for c in df.columns:
        n = int(df[c].isna().sum())
        if not n:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            fixes.append({"issue": f"`{c}` has {n} missing values", "fix": f"Fill `{c}` with its median",
                          "code": f"df['{c}'] = df['{c}'].fillna(df['{c}'].median())"})
        else:
            fixes.append({"issue": f"`{c}` has {n} missing values", "fix": f"Fill `{c}` with 'Unknown'",
                          "code": f"df['{c}'] = df['{c}'].fillna('Unknown')"})
    return fixes


def ai_fixes(df):
    prompt = f"""Profile of a dataset:\n{profile(df).to_string()}\n\nDuplicate rows: {df.duplicated().sum()}
Sample:\n{df.head(8).to_string()}

List the data-quality problems and one fix for each. For each fix give one line of pandas code that reassigns df
(e.g. df = df.drop_duplicates() or df['X'] = ...). Pick sensible strategies (e.g. median for skewed numbers, 'Unknown' for categories,
standardise inconsistent labels, convert text dates with dayfirst=True). Put duplicates first.
Return ONLY a JSON list: [{{"issue": "...", "fix": "...", "code": "..."}}]"""
    reply = llm([{"role": "user", "content": prompt}])
    m = re.search(r"\[.*\]", reply, re.S)
    return json.loads(m.group(0))


def apply_fix(code):
    for b in BLOCKED:
        if b in code:
            raise ValueError(f"Blocked: `{b.strip()}`")
    ns = {"df": st.session_state.df.copy(), "pd": pd, "np": np}
    exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
    st.session_state.df = ns["df"]


# ---------------------------------------------------------------- insights
def offline_insights(df):
    d = df.copy()
    out = []
    if {"Sub_Category", "Profit"} <= set(d.columns):
        p = d.groupby("Sub_Category")["Profit"].sum().sort_values()
        out.append(f"**Loss-maker:** {p.index[0]} lost ₹{abs(p.iloc[0]):,.0f} in total, while the best sub-category, "
                   f"{p.index[-1]}, earned ₹{p.iloc[-1]:,.0f}.")
    if {"Sub_Category", "Discount"} <= set(d.columns):
        disc = d.groupby("Sub_Category")["Discount"].mean().sort_values(ascending=False)
        out.append(f"**Likely cause:** {disc.index[0]} carries the highest average discount ({disc.iloc[0]:.0%}) "
                   f"versus {d['Discount'].mean():.0%} overall, so discounting is eating its margin.")
    if {"Category", "Sales", "Profit"} <= set(d.columns):
        c = d.groupby("Category")[["Sales", "Profit"]].sum()
        c["margin"] = c.Profit / c.Sales
        out.append(f"**Margins differ sharply:** {c.margin.idxmax()} has a {c.margin.max():.1%} profit margin, "
                   f"against {c.margin.min():.1%} for {c.margin.idxmin()}.")
    if "Order_Date" in d.columns and "Sales" in d.columns:
        dt = pd.to_datetime(d["Order_Date"], dayfirst=True, errors="coerce")
        q4 = d[dt.dt.month.isin([10, 11, 12])]["Sales"].mean()
        rest = d[~dt.dt.month.isin([10, 11, 12])]["Sales"].mean()
        out.append(f"**Seasonality:** average order value in Q4 is {q4 / rest - 1:.0%} higher than the rest of the year.")
        y = d.groupby(dt.dt.year)["Sales"].sum()
        if len(y) >= 2:
            out.append(f"**Growth:** sales grew {y.iloc[-1] / y.iloc[-2] - 1:.0%} from {y.index[-2]} to {y.index[-1]}.")
    out.append("**Recommendation:** cap discounts on loss-making sub-categories and review their pricing before Q4, "
               "when volumes peak.")
    return "\n\n".join(f"{i + 1}. {s}" for i, s in enumerate(out))


def ai_insights(df):
    d = df.copy()
    facts = [f"Overall: {len(d)} rows.", d.describe(include="all").T.to_string()]
    cats = [c for c in d.select_dtypes(include=["object", "string"]) if 1 < d[c].nunique() <= 15]
    nums = [c for c in d.select_dtypes("number") if c.lower() not in ("quantity",)][:3]
    for c in cats:
        facts.append(f"\nBy {c}:\n" + d.groupby(c)[nums].agg(["sum", "mean"]).round(2).to_string())
    prompt = ("You are a senior business analyst. Using ONLY the statistics below, write the 5 most important, "
              "non-obvious insights for a manager, each with specific numbers, then 1 actionable recommendation. "
              "Numbered list, bold lead-in per point, no preamble.\n\n" + "\n".join(facts))
    return llm([{"role": "user", "content": prompt}], temperature=0.3)


def credit(task, seconds):
    st.session_state.minutes_saved += MANUAL_MINUTES[task]
    st.session_state.seconds_spent += seconds


# ---------------------------------------------------------------- report
def build_report():
    lines = [f"# AI Data Analysis Report", f"_Dataset: {st.session_state.file_name} · "
             f"Generated {datetime.now():%d %b %Y, %H:%M}_", ""]
    if st.session_state.applied_fixes:
        lines += ["## Data cleaning applied", *[f"- {f}" for f in st.session_state.applied_fixes], ""]
    if st.session_state.history:
        lines.append("## Questions answered")
        for h in st.session_state.history:
            lines += [f"### {h['question']}", h.get("summary") or "", "```python", h["code"], "```", ""]
    if st.session_state.insights:
        lines += ["## Key insights", st.session_state.insights]
    return "\n".join(lines)


# ================================================================= UI
with st.sidebar:
    st.header("⚙️ Settings")
    st.selectbox("AI provider", list(PROVIDERS), key="provider")
    p = PROVIDERS[st.session_state.provider]
    has_env_key = bool(os.getenv(p["key_env"]))
    st.text_input("API key", type="password", key="api_key",
                  placeholder="loaded from .env" if has_env_key else "paste your key")
    st.toggle("Offline demo mode", key="offline", value=False,
              help="Answers only cached questions, with no internet or API needed. Your safety net.")
    st.toggle("Reuse cached answers", key="use_cache", value=True,
              help="Questions you asked during rehearsal replay instantly and identically.")

    cache = load_cache()
    if cache:
        with st.expander(f"📌 Cached questions ({len(cache)})"):
            for v in cache.values():
                st.caption(v["question"])

    st.divider()
    st.subheader("⏱️ Automation tracker")
    c1, c2 = st.columns(2)
    c1.metric("Manual estimate", f"{st.session_state.minutes_saved} min")
    c2.metric("With AI", f"{st.session_state.seconds_spent:.0f} sec")

    if st.session_state.df is not None:
        st.divider()
        st.download_button("⬇️ Cleaned data (CSV)", st.session_state.df.to_csv(index=False),
                           "cleaned_data.csv", width="stretch")
        st.download_button("⬇️ Analysis report (Markdown)", build_report(), "analysis_report.md",
                           width="stretch")

st.title("📊 AI Data Analyst")
st.caption("Generative AI that profiles, cleans, queries and explains your data — and shows its work.")

# ---------------------------------------------------------------- upload
up_col, sample_col = st.columns([3, 1])
uploaded = up_col.file_uploader("Upload a CSV", type="csv", label_visibility="collapsed")
use_sample = sample_col.button("Use sample sales data", width="stretch")


def load(df, name):
    for k in ["history", "applied_fixes"]:
        st.session_state[k] = []
    st.session_state.update(df=df, df_original=df.copy(), file_name=name, fix_suggestions=None,
                            insights=None, minutes_saved=0, seconds_spent=0.0, profiled=False)


if uploaded is not None and uploaded.name != st.session_state.file_name:
    load(pd.read_csv(uploaded), uploaded.name)
if use_sample:
    load(pd.read_csv(APP_DIR / "sales_data.csv"), "sales_data.csv")
    st.rerun()

df = st.session_state.df
if df is None:
    st.info("Upload a CSV or load the sample dataset to begin.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Auto-profile", "2️⃣ AI cleaning", "3️⃣ Ask your data", "4️⃣ Auto-insights"])

# ---------------------------------------------------------------- 1. profile
with tab1:
    t0 = time.time()
    m = st.columns(4)
    m[0].metric("Rows", f"{len(df):,}")
    m[1].metric("Columns", df.shape[1])
    m[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    m[3].metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Column profile")
    st.dataframe(profile(df), width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Numeric summary")
        st.dataframe(df.describe().T.round(2), width="stretch")
    with right:
        num = df.select_dtypes("number").columns.tolist()
        if num:
            col = st.selectbox("Distribution of", num, index=min(len(num) - 1, num.index("Profit") if "Profit" in num else 0))
            st.plotly_chart(px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}"), width="stretch")
    with st.expander("Preview raw data"):
        st.dataframe(df.head(50), width="stretch")
    if not st.session_state.get("profiled"):
        credit("profile", time.time() - t0)
        st.session_state.profiled = True

# ---------------------------------------------------------------- 2. cleaning
with tab2:
    st.write("The AI inspects the profile, finds data-quality problems and proposes a fix for each. "
             "**You approve every change** — nothing is applied automatically.")
    if st.button("🔍 Find data-quality issues", type="primary"):
        t0 = time.time()
        with st.spinner("AI is inspecting the data…"):
            try:
                st.session_state.fix_suggestions = rule_based_fixes(df) if st.session_state.offline else ai_fixes(df)
                st.session_state.fix_source = "rules" if st.session_state.offline else "AI"
            except Exception as e:
                st.warning(f"AI suggestion failed ({e}). Falling back to rule-based checks.")
                st.session_state.fix_suggestions = rule_based_fixes(df)
                st.session_state.fix_source = "rules"
        st.session_state.seconds_spent += time.time() - t0

    fixes = st.session_state.fix_suggestions
    if fixes:
        st.caption(f"Suggestions from: {st.session_state.fix_source}")
        for i, f in enumerate(fixes):
            done = f["fix"] in st.session_state.applied_fixes
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"**Issue:** {f['issue']}  \n**Proposed fix:** {f['fix']}")
                a.code(f["code"], language="python")
                if done:
                    b.success("Applied")
                elif b.button("Apply", key=f"fix{i}"):
                    t0 = time.time()
                    try:
                        apply_fix(f["code"])
                        st.session_state.applied_fixes.append(f["fix"])
                        credit("clean", time.time() - t0)
                        st.rerun()
                    except Exception as e:
                        b.error(str(e))
        if st.session_state.applied_fixes:
            before, after = st.session_state.df_original, st.session_state.df
            st.success(f"Rows: {len(before):,} → {len(after):,} · Missing cells: "
                       f"{int(before.isna().sum().sum())} → {int(after.isna().sum().sum())}")
            if st.button("↩️ Undo all cleaning"):
                st.session_state.df = st.session_state.df_original.copy()
                st.session_state.applied_fixes = []
                st.rerun()

# ---------------------------------------------------------------- 3. ask
with tab3:
    st.write("Ask a question in plain English. The AI writes the pandas code, runs it, and shows the answer, "
             "the chart, and **the exact code it used** so you can verify it.")
    with st.form("ask", clear_on_submit=False):
        q = st.text_input("Your question", placeholder="e.g. Which sub-categories lose money, and how much discount do they get?")
        go_btn = st.form_submit_button("Ask ▶", type="primary")
    if go_btn and q.strip():
        t0 = time.time()
        with st.spinner("Writing and running analysis code…"):
            try:
                out = ask_question(q, df)
                out["question"] = q
                out["seconds"] = time.time() - t0
                st.session_state.history.insert(0, out)
                credit("query", out["seconds"])
            except Exception as e:
                st.error(str(e))

    for i, h in enumerate(st.session_state.history):
        with st.container(border=True):
            st.markdown(f"#### ❓ {h['question']}")
            tag = "⚡ replayed from cache" if h["source"] == "cache" else "🤖 generated live"
            st.caption(f"{tag} · {h['seconds']:.1f}s")
            if h.get("attempts"):
                st.warning(f"First attempt failed and the AI corrected itself "
                           f"({len(h['attempts'])} retry). Error was: `{h['attempts'][0]['error']}`")
            if h.get("summary"):
                st.success(h["summary"])
            if h.get("fig") is not None:
                st.plotly_chart(h["fig"], width="stretch", key=f"fig{i}")
            r = h.get("result")
            if isinstance(r, (pd.DataFrame, pd.Series)):
                st.dataframe(r, width="stretch")
            elif r is not None:
                st.markdown(f"**Result:** {r}")
            with st.expander("🔎 Show the code the AI wrote", expanded=(i == 0)):
                st.code(h["code"], language="python")

# ---------------------------------------------------------------- 4. insights
with tab4:
    st.write("One click: the AI scans every category and metric and writes the findings a manager should know.")
    if st.button("✨ Generate insights", type="primary"):
        t0 = time.time()
        with st.spinner("Analysing every dimension…"):
            try:
                st.session_state.insights = offline_insights(df) if st.session_state.offline else ai_insights(df)
            except Exception as e:
                st.warning(f"AI unavailable ({e}). Showing computed insights instead.")
                st.session_state.insights = offline_insights(df)
        credit("insights", time.time() - t0)
    if st.session_state.insights:
        st.markdown(st.session_state.insights)
        st.info("⚠️ AI-generated insights can sound confident and still be wrong. "
                "Verify any number before it goes into a decision.")
