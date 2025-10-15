#!/usr/bin/env python3
"""
Example usage of the Interview Question Creator
"""

from interview_question_creator import InterviewQuestionCreator

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Initialize the creator
    creator = InterviewQuestionCreator()
    
    # Load the model
    print("Loading model...")
    creator.load_model()
    
    # Load PDFs from data directory
    print("Loading PDFs...")
    texts = creator.load_pdfs_from_directory("data")
    
    if texts:
        print(f"Loaded {len(texts)} documents")
        
        # Generate easy questions
        print("\nGenerating easy questions...")
        easy_questions = creator.generate_questions(
            num_questions=3,
            difficulty="easy",
            question_types=["conceptual"]
        )
        
        if easy_questions:
            print("\nEasy Questions:")
            creator.print_questions(easy_questions)
        
        # Generate hard questions
        print("\nGenerating hard questions...")
        hard_questions = creator.generate_questions(
            num_questions=2,
            difficulty="hard",
            question_types=["analytical"]
        )
        
        if hard_questions:
            print("\nHard Questions:")
            creator.print_questions(hard_questions)
            
        # Save all questions
        all_questions = easy_questions + hard_questions
        creator.save_questions(all_questions, "example_questions.txt")
        print(f"\nAll questions saved to example_questions.txt")
    else:
        print("No documents loaded!")

def example_advanced_usage():
    """Advanced usage example with custom settings"""
    print("\n=== Advanced Usage Example ===")
    
    # Initialize with a different model
    creator = InterviewQuestionCreator(model_name="gpt2")
    
    try:
        # Load the model
        print("Loading GPT-2 model...")
        creator.load_model()
        
        # Load PDFs
        print("Loading PDFs...")
        texts = creator.load_pdfs_from_directory("data")
        
        if texts:
            # Generate mixed difficulty questions
            print("Generating mixed difficulty questions...")
            questions = creator.generate_questions(
                num_questions=8,
                difficulty="medium",
                question_types=["conceptual", "practical", "analytical"]
            )
            
            if questions:
                creator.print_questions(questions)
                creator.save_questions(questions, "advanced_questions.txt")
                print(f"\nQuestions saved to advanced_questions.txt")
        else:
            print("No documents loaded!")
            
    except Exception as e:
        print(f"Error with advanced example: {e}")

if __name__ == "__main__":
    # Run basic example
    example_basic_usage()
    
    # Run advanced example
    example_advanced_usage()
