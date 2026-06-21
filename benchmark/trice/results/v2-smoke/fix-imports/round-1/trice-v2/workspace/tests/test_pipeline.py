def test_import_from_package_root():
    from mypkg import run_pipeline
    assert callable(run_pipeline)

def test_pipeline():
    from mypkg import run_pipeline
    assert run_pipeline([" a ", "", "b\n"]) == ["A", "B"]

def test_submodule_still_importable():
    from mypkg.loaders import read_rows
    assert read_rows(["x "]) == ["x"]
