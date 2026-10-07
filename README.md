# AI Data Analyst
**Generative AI for Data Analysis: Automating Data Analysis Tasks**

Upload a CSV and the app automates the analyst's workflow: it profiles the data, proposes cleaning fixes for you to approve, answers plain-English questions by writing and running pandas code, and writes business insights. Every answer shows the code the AI wrote, so its work can be verified.

## Setup (10 minutes)

1. Install Python 3.10 or newer, then open a terminal in this folder.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Get a free API key from Groq (https://console.groq.com/keys) or Gemini (https://aistudio.google.com/apikey).
4. Rename `.env.example` to `.env` and paste the key after `GROQ_API_KEY=` (or `GEMINI_API_KEY=`).
5. Run the app:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501. Choose the matching provider in the sidebar.

`sales_data.csv` and `demo_cache.json` are already included. `make_data.py` regenerates the dataset and `build_cache.py` re-seeds the cache; you don't need to run either.

## What each tab does

| Tab | What is automated | Syllabus link |
|---|---|---|
| 1. Auto-profile | Shape, types, missing values, duplicates, distributions, instantly on upload | Automating data analysis tasks |
| 2. AI cleaning | AI finds data-quality issues and proposes a one-line fix for each; you approve each one | Automating data analysis tasks |
| 3. Ask your data | Question → AI writes pandas code → app runs it → table, chart, one-line answer, and the code. If the code errors, the error goes back to the AI and it corrects itself once | Generating insights |
| 4. Auto-insights | AI writes the top findings and a recommendation for a manager | Generating insights and recommendations |

The sidebar tracks estimated manual time against actual AI time, and lets you download the cleaned CSV and an auto-written analysis report.

## Safety nets for the live demo

- **Offline demo mode** (sidebar toggle) answers the six rehearsed questions below with no internet or API. Cleaning and insights fall back to rule-based versions. Turn it on the moment the Wi-Fi fails.
- **Reuse cached answers** (on by default) means every question you ask live tonight is saved to `demo_cache.json`, so tomorrow it replays instantly and identically.
- Generated code runs in a restricted sandbox: imports, file access and system calls are blocked.

## Demo script (6–7 minutes)

1. **Hook (30s).** "An analyst spends most of their time cleaning and querying data, not thinking about it. Let's automate that."
2. **Load sample data → Tab 1 (1 min).** Point at 2,040 rows, 132 missing cells and 40 duplicates, found instantly.
3. **Tab 2 (1 min).** Click *Find data-quality issues*. Highlight the inconsistent region labels (north / NORTH / "North "), which would silently split totals. Apply all fixes and show rows 2,040 → 2,000 and missing cells → 0. Stress that a human approves each change.
4. **Tab 3 (2.5 min).** Ask, in order:
   - *What are the total sales and profit by region?*
   - *Show the monthly sales trend* (the Q4 spike appears)
   - *Why is profit negative in Furniture?* (Tables lose ₹19k at a 40% average discount). Open the code expander here.
5. **The deliberate failure (1 min).** Ask *Which region is the best?* The AI confidently answers West, but open the code: it quietly chose total sales. Then ask *Which region is the best by profit margin?* and South wins. Lesson: AI resolves ambiguity silently, so humans must check the logic. This covers responsible AI from Unit I.
6. **Tab 4 (45s).** Click *Generate insights*, then download the report from the sidebar to show automated report writing.
7. **Close (30s).** Point at the time tracker: over an hour of estimated manual Excel work, done in seconds. "GenAI doesn't replace the analyst; it removes the drudgery and leaves the judgement."

## Questions you may be asked

- **Which model does this use?** An open-source Llama model via Groq, or Gemini. The app uses the OpenAI-compatible API, so the provider can be swapped with one setting.
- **Is it safe to run AI-generated code?** It runs in a restricted namespace; imports, file I/O and system calls are blocked before execution. Production systems would use a full container sandbox.
- **Does the data leave the machine?** Only column names, data types, a few sample rows and aggregated statistics are sent to the model, never the full dataset. For confidential data you would use a locally hosted model.
- **Can the AI be wrong?** Yes, which is why it shows its code and why step 5 exists.
