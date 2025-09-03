#!/usr/bin/env python3
import json

# Create a simple notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 1: Foundations & Data Pipeline\n",
                "\n",
                "This notebook demonstrates Phase 1 of our Contract Analysis System."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "print('Phase 1 notebook is ready!')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Contract Analysis (Python 3.11)",
            "language": "python",
            "name": "contract-analysis"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook
with open('notebooks/01_phase1_foundations.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Phase 1 notebook created successfully!")
