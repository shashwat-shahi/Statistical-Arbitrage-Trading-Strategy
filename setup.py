from setuptools import setup, find_packages

setup(
    name="statistical-arbitrage-trading",
    version="1.0.0",
    description="Statistical Arbitrage Trading Strategy using Cointegration and Mean Reversion",
    author="Shashwat Shahi",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
        "statsmodels>=0.13.0",
        "pykalman>=0.9.5",
        "arch>=5.3.0",
        "plotly>=5.3.0",
        "jupyter>=1.0.0",
    ],
    python_requires=">=3.8",
)