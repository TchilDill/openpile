from setuptools import setup
from pathlib import Path


if __name__ == "__main__":
    setup(
        long_description=Path("README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        project_urls={
            "Documentation": "https://openpile.readthedocs.io/en/latest/",
            "Code": "https://github.com/TchilDill/openpile",
            "Issue tracker": "https://github.com/TchilDill/openpile/issues",
        },
    )

    