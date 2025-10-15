import streamlit as st
from typing import List, Dict
from interview_question_creator import InterviewQuestionCreator
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def init_creator() -> InterviewQuestionCreator:
    if 'creator' not in st.session_state:
        st.session_state.creator = InterviewQuestionCreator()
        st.session_state.creator.load_model()
    return st.session_state.creator


def generate_and_display(creator: InterviewQuestionCreator,
                         num_easy: int,
                         num_medium: int,
                         num_hard: int) -> Dict[str, List[Dict[str, str]]]:
    results = {}

    if num_easy > 0:
        results['easy'] = creator.generate_questions(
            num_questions=num_easy,
            difficulty="easy",
            question_types=["conceptual", "practical", "analytical"]
        )

    if num_medium > 0:
        results['medium'] = creator.generate_questions(
            num_questions=num_medium,
            difficulty="medium",
            question_types=["conceptual", "practical", "analytical"]
        )

    if num_hard > 0:
        results['hard'] = creator.generate_questions(
            num_questions=num_hard,
            difficulty="hard",
            question_types=["conceptual", "practical", "analytical"]
        )

    return results


def build_pdf(results: Dict[str, List[Dict[str, str]]]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 50
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, "Interview Questions")
    y -= 24
    c.setFont("Helvetica", 10)

    def draw_wrapped(text: str, max_width: int):
        nonlocal y
        words = text.split(' ')
        line = ""
        for word in words:
            test_line = (line + ' ' + word).strip()
            if c.stringWidth(test_line, "Helvetica", 10) > max_width:
                c.drawString(x_margin, y, line)
                y -= 14
                if y < 60:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 50
                line = word
            else:
                line = test_line
        if line:
            c.drawString(x_margin, y, line)
            y -= 14

    for level in ["easy", "medium", "hard"]:
        questions = results.get(level, [])
        if not questions:
            continue
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_margin, y, f"{level.capitalize()} Questions")
        y -= 18
        c.setFont("Helvetica", 10)
        for idx, q in enumerate(questions, 1):
            draw_wrapped(f"Q{idx}. {q['question']}", width - 2 * x_margin)
            draw_wrapped(f"Answer: {q.get('answer','')}", width - 2 * x_margin)
            y -= 6
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def main():
    st.set_page_config(page_title="Interview Question Creator", page_icon="🧠", layout="wide")
    st.title("🧠 Interview Question Creator")
    st.write("Upload one or more PDF files, then generate interview questions by difficulty.")

    with st.sidebar:
        st.header("Quantities")
        num_easy = st.number_input("Easy questions", min_value=0, max_value=50, value=3)
        num_medium = st.number_input("Medium questions", min_value=0, max_value=50, value=3)
        num_hard = st.number_input("Hard questions", min_value=0, max_value=50, value=3)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Generate Questions", disabled=not uploaded_files):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
            return

        creator = init_creator()

        with st.spinner("Extracting text from PDFs..."):
            pdf_bytes_list = [f.read() for f in uploaded_files]
            texts = creator.load_pdfs_from_uploaded(pdf_bytes_list)

        if not texts:
            st.error("Could not extract text from uploaded PDFs.")
            return

        with st.spinner("Generating questions..."):
            results = generate_and_display(creator, num_easy, num_medium, num_hard)

        if not any(results.values()):
            st.error("No questions generated. Try different settings or model.")
            return

        st.success("Questions generated!")

        tabs = st.tabs(["Easy", "Medium", "Hard"])
        for tab, level in zip(tabs, ["easy", "medium", "hard"]):
            with tab:
                questions = results.get(level, [])
                if not questions:
                    st.info("No questions for this level.")
                    continue
                for idx, q in enumerate(questions, 1):
                    st.markdown(f"**Q{idx}.** {q['question']}")
                    with st.expander("Answer"):
                        ans = (q.get('answer') or '').strip()
                        if not ans:
                            ans = "No answer generated."
                        st.write(ans)

        # Download buttons
        all_lines = []
        for level in ["easy", "medium", "hard"]:
            for q in results.get(level, []):
                all_lines.append(f"[{level.upper()}] {q['question']}")
        all_text = "\n".join(all_lines)
        st.download_button("Download all questions (TXT)", data=all_text, file_name="questions.txt")

        # PDF download
        pdf_bytes = build_pdf(results)
        st.download_button("Download all questions (PDF)", data=pdf_bytes, file_name="questions.pdf", mime="application/pdf")


if __name__ == "__main__":
    main()


