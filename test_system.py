#!/usr/bin/env python3
"""
Test script to verify the Interview Question Creator system
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import PyPDF2
        print("✓ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"✗ PyPDF2 import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    return True

def test_data_directory():
    """Test if data directory and PDFs exist"""
    print("\nTesting data directory...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("✗ Data directory does not exist")
        return False
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("✗ No PDF files found in data directory")
        return False
    
    print(f"✓ Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    return True

def test_pdf_extraction():
    """Test PDF text extraction"""
    print("\nTesting PDF text extraction...")
    
    try:
        from interview_question_creator import InterviewQuestionCreator
        creator = InterviewQuestionCreator()
        
        texts = creator.load_pdfs_from_directory("data")
        if not texts:
            print("✗ No text extracted from PDFs")
            return False
        
        print(f"✓ Successfully extracted text from {len(texts)} documents")
        total_chars = sum(len(text) for text in texts)
        print(f"✓ Total characters extracted: {total_chars}")
        
        return True
        
    except Exception as e:
        print(f"✗ PDF extraction failed: {e}")
        return False

def test_model_loading():
    """Test model loading (without actually loading to save time)"""
    print("\nTesting model configuration...")
    
    try:
        from interview_question_creator import InterviewQuestionCreator
        creator = InterviewQuestionCreator()
        
        # Just test if the class can be instantiated
        print("✓ InterviewQuestionCreator class instantiated successfully")
        
        # Test with a simple model name
        creator_simple = InterviewQuestionCreator(model_name="gpt2")
        print("✓ Can configure different models")
        
        return True
        
    except Exception as e:
        print(f"✗ Model configuration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Interview Question Creator - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Directory Test", test_data_directory),
        ("PDF Extraction Test", test_pdf_extraction),
        ("Model Configuration Test", test_model_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python cli.py --data-dir data --num-questions 3")
        print("2. Or run: python example_usage.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Ensure PDF files exist in the data/ directory")
        print("3. Check that you have sufficient memory for model loading")

if __name__ == "__main__":
    main()
