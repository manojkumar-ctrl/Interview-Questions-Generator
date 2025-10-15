# Interview Question Creator

An AI-powered tool that generates interview questions from PDF documents using Hugging Face transformers and PyPDF.

## Features

-  **PDF Text Extraction**: Automatically extracts text from PDF files using PyPDF
-  **AI Question Generation**: Uses Hugging Face transformers to generate intelligent questions
-  **Multiple Difficulty Levels**: Generate easy, medium, or hard questions
-  **Question Types**: Conceptual, practical, and analytical questions
-  **Export Options**: Save questions to text files
-  **CLI Interface**: Easy-to-use command line interface

## Installation

### 1. Create Virtual Environment

```bash
conda create -n interview python=3.9 -y
conda activate interview
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage - generate 5 medium difficulty questions
python cli.py --data-dir data --num-questions 5

# Generate hard questions with specific types
python cli.py --data-dir data --difficulty hard --types conceptual analytical --num-questions 10

# Use a different model
python cli.py --model microsoft/DialoGPT-large --num-questions 8

# Save to custom file
python cli.py --output my_questions.txt --num-questions 15
```

### Python API

```python
from interview_question_creator import InterviewQuestionCreator

# Initialize creator
creator = InterviewQuestionCreator()

# Load model
creator.load_model()

# Load PDFs
creator.load_pdfs_from_directory("data")

# Generate questions
questions = creator.generate_questions(
    num_questions=5,
    difficulty="medium",
    question_types=["conceptual", "practical"]
)

# Print questions
creator.print_questions(questions)

# Save questions
creator.save_questions(questions, "output.txt")
```

### Example Usage

```bash
# Run the example script
python example_usage.py
```

## Configuration

### Supported Models

- `microsoft/DialoGPT-medium` (default)
- `microsoft/DialoGPT-large`
- `gpt2`
- Any Hugging Face text generation model

### Difficulty Levels

- **Easy**: Basic understanding and definitions
- **Medium**: Moderate complexity with some analysis
- **Hard**: Complex, advanced concepts requiring deep understanding

### Question Types

- **Conceptual**: Focus on understanding concepts and definitions
- **Practical**: Real-world applications and implementation
- **Analytical**: Analysis, comparison, and critical thinking

## File Structure

```
Interview Question Creator/
├── data/                          # PDF files directory
│   ├── AI-UNIT 1.pdf
│   └── Unit-1 (AI).pdf
├── interview_question_creator.py  # Main class
├── cli.py                         # Command line interface
├── example_usage.py              # Usage examples
├── requirements.txt               # Dependencies
└── Readme.md                     # This file
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Try using `gpt2` as a fallback model
2. **PDF Extraction Issues**: Ensure PDFs are not password-protected
3. **Memory Issues**: Use smaller models or reduce text chunk size

### Performance Tips

- Use GPU if available for faster model inference
- Process smaller batches of questions for better quality
- Pre-process PDFs to remove unnecessary formatting

## Dependencies

- `transformers`: Hugging Face transformers library
- `torch`: PyTorch for model inference
- `PyPDF2`: PDF text extraction
- `accelerate`: Model acceleration
- `sentencepiece`: Tokenization
- `protobuf`: Protocol buffers

## License

This project is open source and available under the MIT License.


