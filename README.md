# Financial Fake News Detection

This project investigates the detection of fake news in financial markets using Natural Language Processing (NLP). Misinformation can drastically affect market sentiment and asset prices, making automated detection critical.

## Key Findings

I compared a traditional machine learning approach against a modern deep learning model using a dataset of ~30,000 news articles.

*   **Distinct Patterns Found:** Fake news in this dataset exhibits very different linguistic features compared to real news (which appears to come from established wire services).
*   **Model Performance:**
    *   **Baseline (Logistic Regression):** Achieved **98% accuracy** using simple word-count vectors (TF-IDF).
    *   **DistilBERT (Transformer):** Achieved **100% accuracy**, perfectly distinguishing the deceptive content from legitimate reporting.
*   **Thematic Analysis:** Unsupervised learning (LDA) revealed that the fake news content heavily clusters around specific sensational topics like Middle Eastern conflicts and US political conspiracies, which likely makes it easier for models to flag.

## Project Contents

*   **`Fake_News_Detection_Project.ipynb`**: The main project file. Contains the full code, analysis, visualizations, and the final written report.
*   **`Report.html`**: An export of the notebook for easy reading in any browser.
*   **`topics.txt`**: Raw output of the topic modeling analysis.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Data:**
    The notebook expects data in a `data/` folder. You can run the included helper script:
    ```bash
    python download_data.py
    ```
3.  **Run the Notebook:**
    Launch Jupyter and open `Fake_News_Detection_Project.ipynb`.

