import os
import re
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from io import BytesIO

class InterviewQuestionCreator:
    """
    A class to create interview questions from PDF documents using Hugging Face LLMs.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the Interview Question Creator.
        
        Args:
            model_name (str): Hugging Face model name for question generation
        """
        self.model_name = model_name
        self.text_generator = None
        self.tokenizer = None
        self.model = None
        self.loaded_texts = []
        
    def load_model(self):
        """Load the Hugging Face model and tokenizer."""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Create text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler model
            try:
                self.text_generator = pipeline("text-generation", model="gpt2")
                print("Using fallback model: gpt2")
            except Exception as fallback_error:
                print(f"Fallback model also failed: {fallback_error}")
                raise
    
    def load_pdfs_from_directory(self, directory_path: str) -> List[str]:
        """
        Load and extract text from all PDF files in a directory.
        
        Args:
            directory_path (str): Path to directory containing PDF files
            
        Returns:
            List[str]: List of extracted text from all PDFs
        """
        pdf_texts = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Directory {directory_path} does not exist!")
            return pdf_texts
            
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return pdf_texts
            
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing: {pdf_file.name}")
                text = self.extract_text_from_pdf(str(pdf_file))
                if text.strip():
                    pdf_texts.append(text)
                    print(f"Successfully extracted {len(text)} characters from {pdf_file.name}")
                else:
                    print(f"No text extracted from {pdf_file.name}")
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                
        self.loaded_texts = pdf_texts
        return pdf_texts
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text content
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            
        return text
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF provided as raw bytes. Useful for uploaded files.
        """
        text = ""
        try:
            with BytesIO(pdf_bytes) as byte_stream:
                pdf_reader = PyPDF2.PdfReader(byte_stream)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from bytes: {e}")
        return text

    def load_pdfs_from_filepaths(self, filepaths: List[str]) -> List[str]:
        """
        Load and extract text from specific PDF file paths.
        """
        pdf_texts = []
        for path in filepaths:
            try:
                text = self.extract_text_from_pdf(path)
                if text.strip():
                    pdf_texts.append(text)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        self.loaded_texts = pdf_texts
        return pdf_texts

    def load_pdfs_from_uploaded(self, uploaded_files: List[bytes]) -> List[str]:
        """
        Load and extract text from a list of uploaded PDF files as bytes.
        """
        pdf_texts = []
        for idx, data in enumerate(uploaded_files):
            try:
                text = self.extract_text_from_pdf_bytes(data)
                if text.strip():
                    pdf_texts.append(text)
            except Exception as e:
                print(f"Error processing uploaded file {idx}: {e}")
        self.loaded_texts = pdf_texts
        return pdf_texts

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the extracted text.
        
        Args:
            text (str): Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Split into sentences and filter out very short ones
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return '. '.join(sentences)
    
    def generate_questions(self, 
                          num_questions: int = 5, 
                          difficulty: str = "medium",
                          question_types: List[str] = None) -> List[Dict[str, str]]:
        """
        Generate interview questions based on the loaded text.
        
        Args:
            num_questions (int): Number of questions to generate
            difficulty (str): Difficulty level (easy, medium, hard)
            question_types (List[str]): Types of questions to generate
            
        Returns:
            List[Dict[str, str]]: List of generated questions with metadata
        """
        if not self.loaded_texts:
            print("No text loaded! Please load PDFs first.")
            return []
            
        if not self.text_generator:
            print("Model not loaded! Please load model first.")
            return []
        
        if question_types is None:
            question_types = ["conceptual", "practical", "analytical"]
            
        questions = []
        
        # Combine all loaded texts
        combined_text = " ".join(self.loaded_texts)
        combined_text = self.preprocess_text(combined_text)
        
        # Extract candidate questions directly from the PDF text first
        candidate_questions = self.extract_candidate_questions(combined_text)

        # Split text into chunks for processing
        chunk_size = 1000
        text_chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
        
        print(f"Generating {num_questions} {difficulty} questions...")
        
        for i in range(num_questions):
            try:
                # Select a random chunk
                import random
                chunk = random.choice(text_chunks)

                # Choose a type regardless of question source
                question_type = random.choice(question_types)

                # Prefer PDF-extracted questions; fallback to model-generated
                if i < len(candidate_questions):
                    question_text = candidate_questions[i]
                else:
                    prompt = self.create_question_prompt(chunk, difficulty, question_type)
                    generated = self.text_generator(
                        prompt,
                        max_new_tokens=50,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        truncation=True
                    )
                    question_text = generated[0]['generated_text']
                    question_text = self.clean_generated_question(question_text, prompt)
                question_text = self.refine_question(question_text)
                question_text = self.normalize_question_sentence(question_text)
                answer_text = self.generate_answer(chunk, question_text)
                
                if question_text:
                    questions.append({
                        'question': question_text,
                        'difficulty': difficulty,
                        'type': question_type,
                        'source_chunk': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        'answer': answer_text
                    })
                    
            except Exception as e:
                print(f"Error generating question {i+1}: {e}")
                continue
        
        return questions
    
    def extract_candidate_questions(self, text: str) -> List[str]:
        """
        Extract question-like sentences from the PDF text to ensure questions come from the document.
        Heuristics:
        - Sentences ending with '?'
        - Sentences starting with common interrogatives
        - Deduplicate and clean
        """
        if not text:
            return []
        candidates: List[str] = []
        # Capture sentences that end with a question mark
        for match in re.finditer(r"([^\?]{6,}?\?)", text):
            q = match.group(1).strip()
            candidates.append(q)

        # Also scan for interrogative-starting sentences
        interrogatives = (
            'what', 'why', 'how', 'when', 'where', 'which', 'who', 'whom', 'whose',
            'can', 'could', 'should', 'would', 'is', 'are', 'do', 'does', 'did', 'will', 'shall', 'may', 'might'
        )
        for sentence in re.split(r"(?<=[\.!?])\s+", text):
            s = sentence.strip()
            if not s:
                continue
            if s.lower().startswith(interrogatives):
                if not s.endswith('?'):
                    s = s.rstrip('.!') + '?'
                candidates.append(s)

        # Normalize and dedupe
        cleaned: List[str] = []
        seen = set()
        for q in candidates:
            qn = self.normalize_question_sentence(q)
            if len(qn) < 8 or len(qn) > 220:
                continue
            key = qn.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(qn)

        return cleaned[:200]
    
    def create_question_prompt(self, text_chunk: str, difficulty: str, question_type: str) -> str:
        """Create a prompt for question generation."""
        
        difficulty_prompts = {
            "easy": "Create a simple, basic question about:",
            "medium": "Create a moderately challenging question about:",
            "hard": "Create a complex, advanced question about:"
        }
        
        type_prompts = {
            "conceptual": "Focus on understanding concepts and definitions.",
            "practical": "Focus on real-world applications and implementation.",
            "analytical": "Focus on analysis, comparison, and critical thinking."
        }
        
        base_prompt = f"{difficulty_prompts.get(difficulty, 'Create a question about:')} {text_chunk[:300]}"
        type_instruction = type_prompts.get(question_type, "")
        
        return f"{base_prompt}\n{type_instruction}\nQuestion:"
    
    def clean_generated_question(self, generated_text: str, original_prompt: str) -> str:
        """Clean and extract the question from generated text."""
        # Remove the original prompt
        question = generated_text.replace(original_prompt, "").strip()
        
        # Extract only the question part (before any additional text)
        lines = question.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line or 'what' in line.lower() or 'how' in line.lower() or 'why' in line.lower()):
                return line
                
        return question

    def normalize_question_sentence(self, question: str) -> str:
        """Normalize spacing, capitalization, and ensure it ends with a question mark."""
        q = question.strip()
        q = re.sub(r"\s+", " ", q)
        # Capitalize first letter if alphabetic
        if q and q[0].isalpha():
            q = q[0].upper() + q[1:]
        # Remove trailing punctuation except question mark
        q = q.rstrip()
        if not q.endswith('?'):
            # If it ends with period or missing punctuation, convert to question mark
            q = q.rstrip('.!') + '?'
        return q

    def refine_question(self, question: str) -> str:
        """Use the model to rewrite into a clear, single interrogative sentence."""
        if not self.text_generator:
            return question
        instruction = (
            "Rewrite the following into a single, clear, grammatically correct interview question. "
            "Use neutral tone, avoid quotes, and ensure it ends with a question mark.\n"
            f"Original: {question}\n"
            "Rewritten question:"
        )
        try:
            generated = self.text_generator(
                instruction,
                max_new_tokens=40,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                truncation=True
            )
            rewritten = generated[0]['generated_text']
            # Take text after the cue
            rewritten = rewritten.split('Rewritten question:')[-1].strip()
            rewritten = re.sub(r"[\"\']", "", rewritten)
            return rewritten
        except Exception as e:
            print(f"Error refining question: {e}")
            return question

    def generate_answer(self, source_text: str, question: str) -> str:
        """Generate a concise answer using the same model with a targeted prompt."""
        if not self.text_generator:
            return ""
        instruction = (
            "Based on the following context, provide a concise answer (1-2 sentences) to the question.\n"
            f"Context: {source_text[:500]}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        try:
            generated = self.text_generator(
                instruction,
                max_new_tokens=60,
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True,
                truncation=True
            )
            answer_raw = generated[0]['generated_text']
            answer = answer_raw.split('Answer:')[-1].strip()
            answer = re.sub(r"\s+", " ", answer)
            # Trim to ~2 sentences
            sentences = re.split(r"(?<=[\.!?])\s+", answer)
            answer = ' '.join(sentences[:2]).strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""
    
    def save_questions(self, questions: List[Dict[str, str]], filename: str = "generated_questions.txt"):
        """Save generated questions to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Generated Interview Questions\n")
                f.write("=" * 50 + "\n\n")
                
                for i, q in enumerate(questions, 1):
                    f.write(f"Question {i}:\n")
                    f.write(f"Difficulty: {q['difficulty']}\n")
                    f.write(f"Type: {q['type']}\n")
                    f.write(f"Question: {q['question']}\n")
                    f.write(f"Source: {q['source_chunk']}\n")
                    f.write("-" * 30 + "\n\n")
                    
            print(f"Questions saved to {filename}")
            
        except Exception as e:
            print(f"Error saving questions: {e}")
    
    def print_questions(self, questions: List[Dict[str, str]]):
        """Print questions to console."""
        print("\n" + "="*60)
        print("GENERATED INTERVIEW QUESTIONS")
        print("="*60)
        
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:")
            print(f"Difficulty: {q['difficulty']}")
            print(f"Type: {q['type']}")
            print(f"Question: {q['question']}")
            print("-" * 40)


def main():
    """Main function to demonstrate the Interview Question Creator."""
    print("Interview Question Creator")
    print("=" * 30)
    
    # Initialize the creator
    creator = InterviewQuestionCreator()
    
    # Load the model
    print("Loading Hugging Face model...")
    creator.load_model()
    
    # Load PDFs from data directory
    print("\nLoading PDFs from data directory...")
    texts = creator.load_pdfs_from_directory("data")
    
    if not texts:
        print("No texts loaded. Exiting.")
        return
    
    print(f"Loaded {len(texts)} PDF documents")
    
    # Generate questions
    print("\nGenerating questions...")
    questions = creator.generate_questions(
        num_questions=5,
        difficulty="medium",
        question_types=["conceptual", "practical", "analytical"]
    )
    
    if questions:
        # Print questions
        creator.print_questions(questions)
        
        # Save questions
        creator.save_questions(questions)
    else:
        print("No questions were generated.")


if __name__ == "__main__":
    main()
