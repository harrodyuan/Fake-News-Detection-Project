# Financial Fake News Detection using NLP

This project addresses the critical problem of automated fake news detection in financial markets. Using a dataset of ~30,000 news articles, I compared five different NLP architectures ranging from traditional statistical models to state-of-the-art Transformers.

## Abstract
The efficient operation of financial markets relies on accurate information. This research addresses the automated detection of financial misinformation ("fake news") using Natural Language Processing (NLP). We utilized the "Fake News Detection" dataset (~30,000 articles) to compare multiple classification approaches, ranging from traditional statistical models to modern deep learning architectures. Specifically, we evaluated **Multinomial Naive Bayes**, **Multi-Layer Perceptron (MLP)**, **Convolutional Neural Networks (CNN)**, **Long Short-Term Memory (LSTM)** networks, and a fine-tuned **DistilBERT** transformer. 

Our results highlight the trade-offs between accuracy and computational cost. While all models achieved high accuracy (>96%), the **DistilBERT** model provided the most robust detection (>99.5% accuracy) at the cost of significantly higher training time. The traditional **Naive Bayes** classifier emerged as a highly efficient baseline, offering 96% accuracy with near-instant training. Topic modeling (LDA) further revealed that fake news in this domain heavily clusters around specific sensationalist geopolitical themes.

## Methods

### Data Preprocessing
The raw text data underwent standard preprocessing to reduce noise and standardize the input for non-BERT models:
1.  **Lowercasing**: To ensure "Apple" and "apple" are treated as the same word.
2.  **Noise Removal**: Stripping punctuation and special characters.
3.  **Stopword Removal**: Removing common English words (e.g., "the", "is") using the NLTK library to focus on semantically meaningful content.
4.  **Tokenization**: Splitting text into individual words.

For the **BERT** model, we used the raw, unprocessed text (only tokenized by the specific BERT tokenizer) to preserve sentence structure and punctuation, which carry semantic signals for transformer models.

### Unsupervised Learning: Topic Modeling
To understand the thematic differences between real and fake news, we applied **Latent Dirichlet Allocation (LDA)**. We vectorized the text using TF-IDF and extracted 5 latent topics. This helps visualize whether "fake" news focuses on different subjects compared to "real" news.

### Supervised Learning Models
We formulated the problem as a binary classification task (0 = Fake, 1 = Real). We implemented and compared five distinct architectures:

1.  **Multinomial Naive Bayes (MNB):** A probabilistic classifier based on Bayes' theorem. It assumes independence between features (words). We used **TF-IDF** (Term Frequency-Inverse Document Frequency) vectors as input. This serves as our primary baseline.
2.  **Multi-Layer Perceptron (MLP):** A feedforward artificial neural network. We used a single hidden layer with 100 neurons and ReLU activation on top of TF-IDF vectors. This tests if capturing non-linear relationships between word counts improves performance.
3.  **Convolutional Neural Network (CNN):** A deep learning model typically used for images but effective for text. We used 1D convolutions with varying kernel sizes (2, 3, 4) to capture local n-gram patterns (e.g., "market crash"). Inputs were learned **Word Embeddings**.
4.  **Long Short-Term Memory (LSTM):** A Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequences. We used a **Bidirectional LSTM** to process text forwards and backwards, allowing the model to understand context from the entire sentence.
5.  **DistilBERT (Transformer):** A distilled version of BERT (Bidirectional Encoder Representations from Transformers). This model utilizes **Transfer Learning**, having been pre-trained on a massive corpus (Wikipedia). We fine-tuned it on our dataset to leverage its deep understanding of language context and semantics.

### Evaluation Procedure
The dataset was split into a **Training Set (80%)** and a **Test Set (20%)** using stratified sampling to maintain class balance.
Models were evaluated using **Accuracy**, **Error Rate**, and **Training Time**. We focused on Error Rate to better visualize the improvements at the high end of performance (e.g., reducing error from 4% to 0.5% is an 8x improvement).

## Results

### Unsupervised Learning (LDA)
The LDA analysis identified distinct topics. As seen in the visualizations below, the dataset partitions into clear geopolitical themes (e.g., Middle East conflict, US Politics, European Elections). Interestingly, manual inspection suggests that "Fake" news in this dataset often overly focuses on specific conspiracy-prone topics (like specific geopolitical conflicts), creating a strong thematic signal.

### Supervised Learning Performance
All models performed exceptionally well, indicating that this dataset has strong linguistic signals separating real from fake news. However, the trade-offs are significant.

| Model | Accuracy | Error Rate | Text Representation | Training Time |
|-------|----------|------------|---------------------|---------------|
| **Naive Bayes** | ~96% | ~4.0% | TF-IDF | Instant (<1s) |
| **MLP** | ~98% | ~2.0% | TF-IDF | Fast (~10s) |
| **CNN** | >99% | <1.0% | Learned Embeddings | Moderate (~2 min) |
| **LSTM** | >99% | <1.0% | Learned Embeddings | Slow (~5 min) |
| **DistilBERT** | **>99.5%** | **<0.5%** | Contextual Embeddings | Slowest (Needs GPU) |

*Note: Exact values may vary slightly across runs due to random initialization.*

**Naive Bayes** provided a strong baseline, proving that simple word usage is highly predictive.
**CNN and LSTM** improved upon this by capturing local phrases and sentence structure.
**DistilBERT** achieved near-perfect performance. Its pre-trained knowledge allowed it to handle even the edge cases that confused the simpler models.

## Discussion

The comparison of these five models illustrates the evolution of NLP techniques.

1.  **The Accuracy Ceiling:** While all models show high accuracy (>96%), the difference lies in the **Error Rate**. DistilBERT reduces the error rate of the Naive Bayes model by nearly **8x**. In a high-volume trading environment, this difference is critical—a 4% error rate could lead to thousands of bad trades per day, while 0.5% is much more manageable.
2.  **Efficiency Trade-off:** Naive Bayes is instantaneous. For applications needing real-time processing on low-power devices, it is the clear winner. Deep learning models require significantly more resources.
3.  **Transformer Supremacy:** DistilBERT's near-100% accuracy confirms that for text classification, Transfer Learning is the state-of-the-art. It requires less preprocessing and understands context better than any model trained from scratch.

**Limitations:**
*   **Dataset Ease:** The extremely high accuracy across the board suggests this specific dataset might be "too easy" or contain artifacts (e.g., all fake news coming from one source domain, all real from another) that models exploit. Real-world fake news is often more subtle.
*   **Generalization:** Models trained on this specific political/financial dataset might not generalize to medical fake news or other domains.

**Future Work:**
*   **Adversarial Testing:** We should test these models on subtle "fake" news generated by AI to see if they can detect machine-generated misinformation, which is often more grammatically correct and harder to spot than human-written spam.
*   **Cross-Domain Evaluation:** Testing the model on a completely different fake news dataset (e.g., LIAR dataset) to measure true robustness.

---

## Project Structure

*   **`Fake_News_Detection_Project.ipynb`**: The main project file containing all code, analysis, and visualizations.
*   **`Report.html`**: A viewable export of the notebook for easy reading in the browser.
*   **`data/`**: Directory for the dataset (downloaded via script).
*   **`download_data.py`**: Helper script to download the dataset from Hugging Face.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Data:**
    ```bash
    python download_data.py
    ```
3.  **Run the Notebook:**
    Launch Jupyter and open `Fake_News_Detection_Project.ipynb`.
