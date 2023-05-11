import sys


def test_required_python_version():
    py_major: int = sys.version_info.major
    py_minor: int = sys.version_info.minor
    sys.version_info

    assert py_major == 3
    assert py_minor >= 9
