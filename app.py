"""Zero-shot text classifier built with Hugging Face Transformers + Streamlit.

Assigned task (digit 3): "zero-shot-classification" pipeline.
Personal feature: the user types their own 3 candidate labels.
"""

import pandas as pd
import streamlit as st
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"  # the default model of the zero-shot pipeline
MAX_CHARS = 3000  # BART reads at most 1024 tokens; anything past that is cut off

st.set_page_config(page_title="Zero-Shot Classifier", page_icon="🏷️")

# Mandatory title
st.title("Muhammad Fawwad Ahmed ----------------- B04-0923-000043ma")
st.caption(
    f"Zero-shot text classification with `{MODEL_NAME}`. "
    "Type any text and your own 3 labels – the model has never been trained on them."
)


@st.cache_resource(show_spinner="Loading model (first run downloads ~1.6 GB)…")
def load_classifier():
    return pipeline("zero-shot-classification", model=MODEL_NAME)


@st.cache_data(show_spinner=False)
def classify(text, labels, multi_label, template):
    result = load_classifier()(
        text,
        candidate_labels=list(labels),
        multi_label=multi_label,
        hypothesis_template=template,
    )
    return result["labels"], result["scores"]


# ---------- Input widgets ----------
text = st.text_area(
    "Text to classify",
    value="The new phone has an amazing camera but the battery dies before lunch.",
    height=150,
)

st.subheader("Your 3 candidate labels")
c1, c2, c3 = st.columns(3)
label_1 = c1.text_input("Label 1", value="technology")
label_2 = c2.text_input("Label 2", value="sports")
label_3 = c3.text_input("Label 3", value="politics")

with st.expander("Advanced options"):
    multi_label = st.checkbox(
        "Labels can be true at the same time (multi-label)",
        help="Off: scores add up to 100%. On: each label is scored independently.",
    )
    template = st.text_input("Hypothesis template", value="This text is about {}.")

# ---------- Run ----------
if st.button("Classify", type="primary"):
    labels = [l.strip() for l in (label_1, label_2, label_3)]
    errors = []
    if not text.strip():
        errors.append("Please enter some text to classify.")
    if any(not l for l in labels):
        errors.append("Please fill in all 3 candidate labels.")
    elif len({l.lower() for l in labels}) < 3:
        errors.append("The 3 labels must be different from each other.")
    if "{}" not in template:
        errors.append("The hypothesis template must contain `{}` where the label goes.")

    if errors:
        for e in errors:
            st.warning(e)
    else:
        if len(text) > MAX_CHARS:
            st.info(
                f"Your text is {len(text):,} characters. Only roughly the first "
                f"{MAX_CHARS:,} are read by the model; the rest is ignored."
            )
        try:
            with st.spinner("Classifying…"):
                out_labels, scores = classify(text.strip(), tuple(labels), multi_label, template)
        except Exception as exc:  # network / download / memory problems
            st.error(f"The model could not run: {exc}")
        else:
            top, top_score = out_labels[0], scores[0]
            st.success(f"Top label: **{top}** ({top_score:.1%})")
            if not multi_label and top_score < 0.5:
                st.warning(
                    "Low confidence – the text may not fit any of your labels well. "
                    "Zero-shot always picks one, so treat this result with caution."
                )

            df = pd.DataFrame({"Label": out_labels, "Score": scores})
            st.bar_chart(df, x="Label", y="Score", horizontal=True)
            st.dataframe(
                df.style.format({"Score": "{:.2%}"}), hide_index=True, width="stretch"
            )

st.divider()
st.caption(
    "⚠️ AI predictions can be wrong or biased. Do not use this app for decisions "
    "about people (hiring, grading, moderation) without human review."
)
