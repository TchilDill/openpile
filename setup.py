from setuptools import setup
from pathlib import Path


if __name__ == "__main__":
    setup(
        long_description=Path("README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
    )

    