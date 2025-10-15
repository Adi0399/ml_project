def test_import():
    import ml_project
    assert hasattr(ml_project, "__version__") is False
