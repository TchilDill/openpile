from setuptools import setup
from pathlib import Path
import re

if __name__ == "__main__":
    version = re.findall(
        r'__version__ = "(\d+.\d+.\d+[^"]*)"',
        Path("src/openpile/__init__.py").read_text(encoding="utf-8"),
    )[0]

    setup(
        version=version,
        long_description=Path("README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        project_urls={
            "Documentation": "https://openpile.readthedocs.io/en/latest/",
            "Code": "https://github.com/TchilDill/openpile",
            "Issue tracker": "https://github.com/TchilDill/openpile/issues",
        },
    )
