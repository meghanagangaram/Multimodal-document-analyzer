from __future__ import annotations

import io
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Multimodal Document Analyzer",
    page_icon="📄",
    layout="wide",
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}

CLASSIFICATION_KEYWORDS = {
    "Invoice / Receipt": [
        "invoice",
        "receipt",
        "subtotal",
        "tax",
        "amount due",
        "payment",
        "bill to",
    ],
    "Contract / Legal": [
        "agreement",
        "contract",
        "party",
        "clause",
        "terms",
        "liability",
        "confidential",
    ],
    "Research / Academic": [
        "abstract",
        "methodology",
        "references",
        "experiment",
        "dataset",
        "results",
        "citation",
    ],
    "Financial Report": [
        "revenue",
        "profit",
        "loss",
        "assets",
        "liabilities",
        "quarter",
        "balance sheet",
    ],
    "Medical / Health": [
        "patient",
        "diagnosis",
        "treatment",
        "clinical",
        "symptoms",
        "prescription",
        "medical",
    ],
    "Resume / Profile": [
        "experience",
        "education",
        "skills",
        "projects",
        "employment",
        "linkedin",
        "resume",
    ],
}


@dataclass
class PageAnalysis:
    page_number: int
    text: str = ""
    ocr_text: str = ""
    tables: list[pd.DataFrame] = field(default_factory=list)
    chart_count: int = 0
    handwriting_hint: bool = False


@dataclass
class DocumentAnalysis:
    name: str
    file_type: str
    text: str
    pages: list[PageAnalysis] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    chart_count: int = 0
    handwriting_hints: int = 0


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def keyword_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        if token.lower() not in STOPWORDS
    ]


def extract_text_from_image(image: Image.Image) -> str:
    if pytesseract is None:
        return ""
    prepared = ImageOps.grayscale(image)
    try:
        return pytesseract.image_to_string(prepared)
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def has_tesseract_binary() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def detect_visual_elements(image: Image.Image) -> tuple[int, bool]:
    if cv2 is None:
        return 0, False

    array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=max(30, image.width // 12),
        maxLineGap=8,
    )

    chart_count = 0
    if lines is not None:
        horizontal = 0
        vertical = 0
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            if abs(y1 - y2) < 6:
                horizontal += 1
            if abs(x1 - x2) < 6:
                vertical += 1
        chart_count = 1 if horizontal >= 4 and vertical >= 2 else 0

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_curvy_marks = sum(1 for contour in contours if 15 < cv2.arcLength(contour, True) < 220)
    handwriting_hint = small_curvy_marks > 150 and chart_count == 0
    return chart_count, handwriting_hint


def extract_tables_from_text(text: str) -> list[pd.DataFrame]:
    tables = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = [line for line in lines if re.search(r"\s{2,}|\t|\|", line)]
    if len(candidates) < 2:
        return tables

    rows = []
    for line in candidates:
        if "|" in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
        elif "\t" in line:
            parts = [part.strip() for part in line.split("\t")]
        else:
            parts = [part.strip() for part in re.split(r"\s{2,}", line)]
        if len(parts) >= 2:
            rows.append(parts)

    if len(rows) >= 2:
        max_cols = max(len(row) for row in rows)
        padded = [row + [""] * (max_cols - len(row)) for row in rows]
        tables.append(make_safe_dataframe(padded[1:], padded[0]))
    return tables


def unique_column_names(columns: list[Any]) -> list[str]:
    names = []
    seen: dict[str, int] = {}
    for index, column in enumerate(columns, start=1):
        base = clean_text(str(column or "")) or f"Column {index}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base} {count + 1}")
    return names


def make_safe_dataframe(rows: list[list[Any]], columns: list[Any] | None = None) -> pd.DataFrame:
    if columns is None:
        max_cols = max((len(row) for row in rows), default=0)
        columns = [f"Column {index}" for index in range(1, max_cols + 1)]

    safe_columns = unique_column_names(list(columns))
    padded_rows = [row + [""] * (len(safe_columns) - len(row)) for row in rows]
    return pd.DataFrame(padded_rows, columns=safe_columns)


def safe_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    display_table = table.copy()
    display_table.columns = unique_column_names(list(display_table.columns))
    return display_table


def read_pdf(uploaded_file: Any) -> DocumentAnalysis:
    data = uploaded_file.getvalue()
    analysis = DocumentAnalysis(uploaded_file.name, "PDF", "")

    if fitz is not None:
        doc = fitz.open(stream=data, filetype="pdf")
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            page_analysis = PageAnalysis(page_number=index, text=text)

            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            analysis.images.append(image)

            if not clean_text(text):
                page_analysis.ocr_text = extract_text_from_image(image)

            chart_count, handwriting_hint = detect_visual_elements(image)
            page_analysis.chart_count = chart_count
            page_analysis.handwriting_hint = handwriting_hint
            analysis.pages.append(page_analysis)
    else:
        st.warning("PyMuPDF is not installed, so PDF page rendering and OCR fallback are unavailable.")

    if pdfplumber is not None:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for index, page in enumerate(pdf.pages):
                if index >= len(analysis.pages):
                    analysis.pages.append(PageAnalysis(page_number=index + 1))
                text = page.extract_text() or analysis.pages[index].text
                analysis.pages[index].text = text or ""

                for raw_table in page.extract_tables() or []:
                    if raw_table and len(raw_table) > 1:
                        header = [str(cell or "").strip() for cell in raw_table[0]]
                        rows = [[str(cell or "").strip() for cell in row] for row in raw_table[1:]]
                        table = make_safe_dataframe(rows, header)
                        analysis.pages[index].tables.append(table)
                        analysis.tables.append(table)
    elif analysis.pages:
        st.info("Install pdfplumber for stronger native PDF table extraction.")

    combined_text = "\n\n".join(
        clean_text(page.text) or clean_text(page.ocr_text) for page in analysis.pages
    )
    analysis.text = combined_text
    analysis.chart_count = sum(page.chart_count for page in analysis.pages)
    analysis.handwriting_hints = sum(1 for page in analysis.pages if page.handwriting_hint)

    if not analysis.tables:
        analysis.tables.extend(extract_tables_from_text(combined_text))

    return analysis


def read_image(uploaded_file: Any) -> DocumentAnalysis:
    image = Image.open(uploaded_file).convert("RGB")
    ocr_text = extract_text_from_image(image)
    chart_count, handwriting_hint = detect_visual_elements(image)
    page = PageAnalysis(
        page_number=1,
        text="",
        ocr_text=ocr_text,
        chart_count=chart_count,
        handwriting_hint=handwriting_hint,
    )
    return DocumentAnalysis(
        name=uploaded_file.name,
        file_type="Image / Scan",
        text=ocr_text,
        pages=[page],
        images=[image],
        tables=extract_tables_from_text(ocr_text),
        chart_count=chart_count,
        handwriting_hints=int(handwriting_hint),
    )


def read_text_file(uploaded_file: Any) -> DocumentAnalysis:
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    return DocumentAnalysis(
        name=uploaded_file.name,
        file_type="Text",
        text=text,
        pages=[PageAnalysis(page_number=1, text=text)],
        tables=extract_tables_from_text(text),
    )


def analyze_file(uploaded_file: Any) -> DocumentAnalysis:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix == ".pdf":
        return read_pdf(uploaded_file)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return read_image(uploaded_file)
    return read_text_file(uploaded_file)


def summarize(text: str, max_sentences: int = 5) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    centroid = np.asarray(matrix.mean(axis=0))
    scores = cosine_similarity(matrix, centroid).ravel()
    selected = sorted(np.argsort(scores)[-max_sentences:])
    return [sentences[index] for index in selected]


def extract_entities(text: str) -> dict[str, list[str]]:
    entities = {
        "Emails": sorted(set(re.findall(r"[\w.\-+]+@[\w.\-]+\.\w+", text))),
        "Phone Numbers": sorted(set(re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)", text))),
        "Dates": sorted(
            set(
                re.findall(
                    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
                    r"[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
                    text,
                    flags=re.IGNORECASE,
                )
            )
        ),
        "Money": sorted(set(re.findall(r"(?:[$₹€£]\s?\d[\d,]*(?:\.\d+)?)", text))),
        "Organizations / Proper Nouns": [],
    }
    proper_nouns = re.findall(r"\b(?:[A-Z][a-zA-Z&.\-]+(?:\s+|$)){2,}", text)
    entities["Organizations / Proper Nouns"] = sorted(
        {clean_text(noun) for noun in proper_nouns if len(clean_text(noun)) > 3}
    )[:30]
    return entities


def classify_document(text: str) -> tuple[str, dict[str, int]]:
    lowered = text.lower()
    scores = {}
    for label, keywords in CLASSIFICATION_KEYWORDS.items():
        scores[label] = sum(lowered.count(keyword) for keyword in keywords)
    best_label = max(scores, key=scores.get) if scores else "General Document"
    if scores.get(best_label, 0) == 0:
        best_label = "General Document"
    return best_label, scores


def detect_topics(text: str, topic_count: int = 4) -> list[list[str]]:
    tokens = keyword_tokens(text)
    if len(tokens) < 20:
        return [keyword for keyword, _ in Counter(tokens).most_common(10)]

    chunks = split_sentences(text)
    if len(chunks) < topic_count:
        chunks = [" ".join(tokens[index : index + 60]) for index in range(0, len(tokens), 60)]

    topic_count = max(1, min(topic_count, len(chunks)))
    vectorizer = CountVectorizer(stop_words="english", max_features=800)
    try:
        matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        return [keyword for keyword, _ in Counter(tokens).most_common(10)]
    lda = LatentDirichletAllocation(n_components=topic_count, random_state=42)
    lda.fit(matrix)
    words = np.array(vectorizer.get_feature_names_out())
    return [words[topic.argsort()[-6:][::-1]].tolist() for topic in lda.components_]


def key_insights(analysis: DocumentAnalysis) -> list[str]:
    words = keyword_tokens(analysis.text)
    label, _ = classify_document(analysis.text)
    insights = [
        f"Document type appears to be: {label}.",
        f"Processed {len(analysis.pages)} page(s) with about {len(analysis.text.split()):,} extracted words.",
    ]
    if analysis.tables:
        insights.append(f"Found {len(analysis.tables)} possible table(s).")
    if analysis.chart_count:
        insights.append(f"Detected {analysis.chart_count} page/image region(s) that may contain charts or structured plots.")
    if analysis.handwriting_hints:
        insights.append(f"Found handwriting-like visual patterns on {analysis.handwriting_hints} page(s).")
    if words:
        top_terms = ", ".join(term for term, _ in Counter(words).most_common(8))
        insights.append(f"Most prominent terms: {top_terms}.")
    return insights


def make_search_index(documents: list[DocumentAnalysis]) -> tuple[list[dict[str, Any]], Any, Any]:
    chunks = []
    for doc_index, document in enumerate(documents):
        for page in document.pages:
            page_text = clean_text(page.text) or clean_text(page.ocr_text)
            sentences = split_sentences(page_text) or [page_text]
            for start in range(0, len(sentences), 4):
                chunk = clean_text(" ".join(sentences[start : start + 4]))
                if chunk:
                    chunks.append(
                        {
                            "doc_index": doc_index,
                            "document": document.name,
                            "page": page.page_number,
                            "text": chunk,
                        }
                    )

    if not chunks:
        return [], None, None

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return chunks, vectorizer, matrix


def search_documents(
    query: str,
    chunks: list[dict[str, Any]],
    vectorizer: Any,
    matrix: Any,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not query or vectorizer is None:
        return []
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = np.argsort(scores)[::-1][:limit]
    return [{**chunks[index], "score": float(scores[index])} for index in ranked if scores[index] > 0]


def answer_question(query: str, documents: list[DocumentAnalysis]) -> tuple[str, list[dict[str, Any]]]:
    chunks, vectorizer, matrix = make_search_index(documents)
    results = search_documents(query, chunks, vectorizer, matrix, limit=4)
    if not results:
        return "I could not find enough relevant evidence in the uploaded documents.", []

    evidence_text = " ".join(result["text"] for result in results)
    summary = summarize(evidence_text, max_sentences=3)
    answer = " ".join(summary) if summary else results[0]["text"]
    return answer, results


def cluster_documents(documents: list[DocumentAnalysis]) -> dict[int, list[str]]:
    usable = [document for document in documents if clean_text(document.text)]
    if len(usable) < 2:
        return {}
    cluster_count = min(4, len(usable))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1200)
    matrix = vectorizer.fit_transform([document.text for document in usable])
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(matrix)
    clusters = defaultdict(list)
    for label, document in zip(labels, usable):
        clusters[int(label) + 1].append(document.name)
    return dict(clusters)


def render_document_card(document: DocumentAnalysis) -> None:
    with st.expander(document.name, expanded=False):
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Summary")
            summary = summarize(document.text)
            if summary:
                for sentence in summary:
                    st.write(f"- {sentence}")
            else:
                st.write("No readable text was extracted.")

            st.subheader("Key Insights")
            for insight in key_insights(document):
                st.write(f"- {insight}")

        with right:
            label, scores = classify_document(document.text)
            st.metric("Predicted Class", label)
            st.metric("Extracted Words", f"{len(document.text.split()):,}")
            st.metric("Tables", len(document.tables))
            st.metric("Chart Hints", document.chart_count)

            score_frame = pd.DataFrame(
                [{"Class": key, "Score": value} for key, value in scores.items()]
            ).sort_values("Score", ascending=False)
            st.bar_chart(score_frame, x="Class", y="Score", height=240)

        st.subheader("Entities")
        entities = extract_entities(document.text)
        entity_cols = st.columns(3)
        for index, (label, values) in enumerate(entities.items()):
            with entity_cols[index % 3]:
                st.caption(label)
                if values:
                    st.write(", ".join(values[:12]))
                else:
                    st.write("None found")

        st.subheader("Topics")
        topics = detect_topics(document.text)
        if topics and isinstance(topics[0], list):
            for index, topic in enumerate(topics, start=1):
                st.write(f"Topic {index}: {', '.join(topic)}")
        elif topics:
            st.write(", ".join(topics))
        else:
            st.write("Not enough text for topic detection.")

        if document.tables:
            st.subheader("Extracted Tables")
            for index, table in enumerate(document.tables, start=1):
                st.caption(f"Table {index}")
                st.dataframe(safe_table_for_display(table), use_container_width=True)

        if document.images:
            st.subheader("Page / Image Preview")
            preview_cols = st.columns(min(3, len(document.images)))
            for index, image in enumerate(document.images[:6]):
                with preview_cols[index % len(preview_cols)]:
                    st.image(image, caption=f"Page/Image {index + 1}", use_container_width=True)

        st.subheader("Extracted Text")
        st.text_area(
            "Full text",
            value=document.text[:20000],
            height=220,
            key=f"text-{document.name}",
        )


def main() -> None:
    st.title("AI-Powered Multimodal Document Analyzer")
    st.caption(
        "Upload PDFs, scans, images, and text files to extract text, tables, entities, "
        "topics, summaries, visual hints, search results, and answers."
    )

    with st.sidebar:
        st.header("Upload")
        files = st.file_uploader(
            "Documents",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "txt", "md", "csv"],
            accept_multiple_files=True,
        )
        st.divider()
        st.header("Runtime")
        st.write(f"OCR: {'available' if has_tesseract_binary() else 'not available'}")
        st.write(f"PDF tables: {'available' if pdfplumber else 'not installed'}")
        st.write(f"Visual detection: {'available' if cv2 else 'not installed'}")
        st.info(
            "For scanned PDFs and handwriting, install Tesseract OCR on your system and keep "
            "pytesseract in the Python environment."
        )

    if not files:
        st.info("Upload one or more documents to begin.")
        return

    cache_key = tuple((file.name, file.size) for file in files)
    if st.session_state.get("cache_key") != cache_key:
        with st.spinner("Extracting and analyzing documents..."):
            st.session_state.documents = [analyze_file(file) for file in files]
            st.session_state.cache_key = cache_key

    documents: list[DocumentAnalysis] = st.session_state.documents
    total_words = sum(len(document.text.split()) for document in documents)
    total_tables = sum(len(document.tables) for document in documents)
    total_charts = sum(document.chart_count for document in documents)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Documents", len(documents))
    metric_cols[1].metric("Words", f"{total_words:,}")
    metric_cols[2].metric("Tables", total_tables)
    metric_cols[3].metric("Chart Hints", total_charts)

    tab_overview, tab_search, tab_qa, tab_documents = st.tabs(
        ["Overview", "Search", "Question Answering", "Document Details"]
    )

    with tab_overview:
        st.subheader("Corpus Summary")
        corpus_text = "\n\n".join(document.text for document in documents)
        for sentence in summarize(corpus_text, max_sentences=7):
            st.write(f"- {sentence}")

        st.subheader("Corpus Topics")
        topics = detect_topics(corpus_text, topic_count=5)
        if topics and isinstance(topics[0], list):
            topic_frame = pd.DataFrame(
                [{"Topic": f"Topic {index}", "Keywords": ", ".join(topic)} for index, topic in enumerate(topics, start=1)]
            )
            st.dataframe(topic_frame, use_container_width=True, hide_index=True)

        clusters = cluster_documents(documents)
        if clusters:
            st.subheader("Similar Document Groups")
            for label, names in clusters.items():
                st.write(f"Group {label}: {', '.join(names)}")

    with tab_search:
        st.subheader("Intelligent Search")
        query = st.text_input("Search across uploaded documents", key="search")
        chunks, vectorizer, matrix = make_search_index(documents)
        results = search_documents(query, chunks, vectorizer, matrix)
        for result in results:
            st.markdown(
                f"**{result['document']}** · page {result['page']} · relevance {result['score']:.2f}"
            )
            st.write(result["text"])
            st.divider()
        if query and not results:
            st.write("No matching passages found.")

    with tab_qa:
        st.subheader("Ask a Question")
        question = st.text_input("Question", placeholder="Example: What are the main risks or conclusions?")
        if question:
            answer, evidence = answer_question(question, documents)
            st.write(answer)
            if evidence:
                st.caption("Evidence")
                for result in evidence:
                    st.markdown(f"**{result['document']}**, page {result['page']}")
                    st.write(result["text"])

    with tab_documents:
        for document in documents:
            render_document_card(document)


if __name__ == "__main__":
    main()
