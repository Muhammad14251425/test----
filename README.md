# Zero-Shot Text Classifier (Hugging Face + Streamlit)

**Muhammad Fawwad Ahmed – B04-0923-000043ma**
Assigned digit **3** → `zero-shot-classification` pipeline.
Personal feature: **the user types their own 3 candidate labels.**

AI assistant used while building: Claude Code (Opus 5.5).

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (Task 2.2) |
| `requirements.txt` | Dependencies for Streamlit Cloud / local runs |
| `task_2_1_test_model.py` | Load the pipeline and run one quick test (Task 2.1) |
| `Task2_colab.ipynb` | Colab notebook: Task 2.1 test + running the app from Colab (Task 2.3) |
| `hf_space/` | Static Hugging Face Space that embeds the deployed Streamlit app (Task 2.4) |

---

## Task 2.1 – Set up and load the model

**Exact model:** [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli)
(BART-large, ~407M parameters, fine-tuned on the MultiNLI dataset). It is the default model of
`pipeline("zero-shot-classification")`; the code names it explicitly so the model can never change silently.

How it works: each candidate label is turned into a sentence ("This text is about *technology*.")
and the NLI model scores whether the input text *entails* that sentence. The entailment scores
are turned into probabilities over the labels.

Steps in Google Colab (free):

1. Open `Task2_colab.ipynb` in Colab (File ▸ Upload notebook).
2. Run `!pip install -q transformers torch streamlit`.
3. Run the load cell:
   ```python
   from transformers import pipeline
   classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
   ```
4. Run the quick-test cell and **screenshot the output** (labels with scores; `technology` should come out on top).

## Task 2.2 – The Streamlit app (`app.py`)

| Requirement | Where in `app.py` |
|---|---|
| Title with name & ID | `st.title("Muhammad Fawwad Ahmed ----------------- B04-0923-000043ma")` |
| Input widget | `st.text_area` for the text |
| Personal feature | three `st.text_input` boxes for the user's own 3 labels |
| Button | `st.button("Classify")` |
| Cached model call | `@st.cache_resource` loads the model once; `@st.cache_data` caches repeated identical predictions |
| Result display | success box with top label, horizontal bar chart, and a score table |
| Extras | multi-label toggle, editable hypothesis template, input validation, low-confidence warning, long-text notice, error handling, responsible-AI disclaimer |

## Task 2.3 – Run locally and test

Local run (or use the last cells of the Colab notebook, which give a public `trycloudflare.com` link):

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Open http://localhost:8501. The first click downloads the model (~1.6 GB) once.

Take two screenshots: **(1)** the normal input below, **(2)** one edge case.

### Test table

Fill the *Actual result* column from your own run.

| # | Type | Input text | Labels | Expected behaviour | Actual result | Pass? |
|---|---|---|---|---|---|---|
| 1 | Normal | "The new phone has an amazing camera but the battery dies before lunch." | technology / sports / politics | Top label **technology** with high score | | |
| 2 | Normal | "Messi scored twice as Inter Miami won the final." | sports / finance / cooking | Top label **sports** | | |
| 3 | Edge – empty | *(empty / only spaces)* | any | Warning "Please enter some text to classify.", model not called | | |
| 4 | Edge – missing label | any text | technology / *(empty)* / politics | Warning "Please fill in all 3 candidate labels." | | |
| 5 | Edge – duplicate labels | any text | Sports / sports / music | Warning "The 3 labels must be different from each other." | | |
| 6 | Edge – very long text | a ~6,000-character paste | any 3 | Info note that only ~the first 3,000 characters are read; still returns a result | | |
| 7 | Edge – tricky (none fits) | "I had pasta for dinner." | technology / sports / politics | Still picks one label, but shows the **low-confidence** warning | | |
| 8 | Edge – tricky (sarcasm) | "Oh great, another Monday. I just *love* traffic." | happy / angry / neutral | Model may wrongly pick *happy* – shows a real limitation | | |

## Task 2.4 – Deployment (Streamlit Community Cloud → embedded in a Hugging Face Space)

**A. Deploy the app on Streamlit Community Cloud (free)**

1. Push `app.py` and `requirements.txt` to a public GitHub repository (this repo).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app ▸ Deploy a public app from GitHub**.
4. Repository: `muhammad14251425/test----`, branch: the branch with `app.py` (e.g. `main`),
   main file path: `app.py`. Optionally pick a custom subdomain, e.g. `fawwad-zero-shot`.
5. Click **Deploy**. The first build installs torch and downloads the model, so wait a few minutes.
6. Copy the URL, e.g. `https://fawwad-zero-shot.streamlit.app`.

**B. Embed it in a Hugging Face Space (static HTML)**

1. Sign in at https://huggingface.co ▸ **New ▸ Space**.
2. Name it (e.g. `zero-shot-classifier`), choose **SDK: Static**, template **Blank**, hardware **CPU basic (free)**, visibility **Public**, and create it.
3. In `hf_space/index.html`, replace both `STREAMLIT_APP_URL` with your Streamlit URL.
4. In the Space, open **Files ▸ Add file ▸ Upload files** and upload `hf_space/index.html` and `hf_space/README.md`
   (it replaces the default ones), then commit.
   Or with git:
   ```bash
   git clone https://huggingface.co/spaces/<your-hf-username>/zero-shot-classifier
   ```
   copy the two files in, then `git add . && git commit -m "Embed Streamlit app" && git push`
   (use a Hugging Face access token with write rights as the password).
5. Open `https://huggingface.co/spaces/<your-hf-username>/zero-shot-classifier`; the Streamlit app shows inside the page.

Note: free Streamlit apps go to sleep after a period of no traffic. If the embed shows
"This app has gone to sleep", open the app link once and click **Yes, get this app back up**.

*Alternative:* run the app directly on Hugging Face with **Create new Space ▸ Docker ▸ Streamlit template**,
then replace its `src/streamlit_app.py` with `app.py` and add `transformers`, `torch` to its `requirements.txt`.

## Task 2.5 – Reflect and critique

*(Draft based on what actually happened while building; rewrite it in your own words.)*

**(a) Something that broke and how I fixed it.**
When I first ran the app in the build environment, the model would not load: the network blocked
`huggingface.co`, so `pipeline()` failed to download `facebook/bart-large-mnli` and the whole Streamlit
page crashed with a red traceback. I fixed it in two ways: I wrapped the model call in `try/except` so the
app shows a clear "The model could not run" message instead of crashing, and I moved the actual model run to
Google Colab, where Hugging Face downloads work. A second, smaller problem was that Streamlit warned that
`use_container_width` is deprecated, so I changed it to `width="stretch"`.

**(b) One real limitation/risk and one responsible-AI safeguard.**
*Limitation:* zero-shot classification **always** picks one of the three labels, even when none of them
fits ("I had pasta for dinner" still gets labelled *technology*, *sports* or *politics*), and because the
scores add up to 100 % the answer can look more confident than it really is. It also struggles with
sarcasm, reads only English well, and silently ignores text beyond ~1,024 tokens. If someone used it to
sort complaints or flag people's posts, those wrong but confident-looking labels could cause unfair decisions.
*Safeguard:* a confidence threshold with a "not sure" outcome – the app already warns when the top score is
below 50 %; in a real deployment I would turn that into an automatic **"Needs human review"** result
(no label is applied), plus keep the on-page disclaimer that the app must not be used for decisions about people
without human review.
