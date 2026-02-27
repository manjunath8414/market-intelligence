from setuptools import setup, find_packages

setup(
    name="market-intelligence",
    version="1.0.0",
    description="Institutional Market Intelligence & Trade Decision Platform for Indian Stocks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.31",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "mplfinance>=0.12.10b0",
        "plotly>=5.15.0",
        "ta>=0.11.0",
        "requests>=2.31.0",
        "rich>=13.5.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "market-intel=market_intelligence.main:cli",
        ],
    },
)
