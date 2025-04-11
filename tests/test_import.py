"""
Test that the package can be imported correctly.
"""

def test_import():
    """Test that the package can be imported."""
    import openopticalflow
    assert hasattr(openopticalflow, "__version__")
