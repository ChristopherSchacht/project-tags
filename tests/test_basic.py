def test_imports():
    """Test that all main modules can be imported."""
    try:
        from modules.pdf_processor import PDFProcessor
        from modules.text_analyzer import TextAnalyzer
        from modules.ai_handler import AIHandler
        assert True
    except ImportError as e:
        assert False, f"Import failed: {str(e)}"
