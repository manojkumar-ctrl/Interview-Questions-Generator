#!/usr/bin/env python3
"""
Command Line Interface for Interview Question Creator
"""

import argparse
import sys
from pathlib import Path
from interview_question_creator import InterviewQuestionCreator

def main():
    parser = argparse.ArgumentParser(
        description="Generate interview questions from PDF documents using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --data-dir data --num-questions 10 --difficulty hard
  python cli.py --data-dir data --output questions.txt --types conceptual practical
  python cli.py --model microsoft/DialoGPT-large --num-questions 5
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing PDF files (default: data)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions to generate (default: 5)"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty level of questions (default: medium)"
    )
    
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["conceptual", "practical", "analytical"],
        default=["conceptual", "practical", "analytical"],
        help="Types of questions to generate (default: all types)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Hugging Face model name (default: microsoft/DialoGPT-medium)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="generated_questions.txt",
        help="Output file for questions (default: generated_questions.txt)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        sys.exit(1)
    
    # Check for PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{args.data_dir}'!")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(pdf_files)} PDF files in {args.data_dir}")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file.name}")
    
    try:
        # Initialize the creator
        print("Initializing Interview Question Creator...")
        creator = InterviewQuestionCreator(model_name=args.model)
        
        # Load the model
        print("Loading AI model...")
        creator.load_model()
        
        # Load PDFs
        print(f"Loading PDFs from {args.data_dir}...")
        texts = creator.load_pdfs_from_directory(str(data_dir))
        
        if not texts:
            print("Error: No text could be extracted from PDF files!")
            sys.exit(1)
        
        print(f"Successfully loaded {len(texts)} documents")
        
        # Generate questions
        print(f"Generating {args.num_questions} {args.difficulty} questions...")
        questions = creator.generate_questions(
            num_questions=args.num_questions,
            difficulty=args.difficulty,
            question_types=args.types
        )
        
        if not questions:
            print("Error: No questions were generated!")
            sys.exit(1)
        
        # Display results
        print(f"\nSuccessfully generated {len(questions)} questions!")
        creator.print_questions(questions)
        
        # Save to file
        creator.save_questions(questions, args.output)
        print(f"\nQuestions saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
