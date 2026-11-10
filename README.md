# Fake News Prediction

A machine learning project that classifies a news article as **Fake
News** or **Not a Fake News** using its text.

The project is implemented in a Jupyter Notebook
(`FakeNewsPrediction(1).ipynb`) with Python and scikit-learn. It
preprocesses news text, converts it into numerical TF-IDF features,
trains four classification models, and compares their predictions on a
test set.

## Project workflow

1.  Load the fake and true news datasets from `Fake.csv` and `True.csv`.
2.  Add class labels:
    -   `0` = Fake News
    -   `1` = Not a Fake News
3.  Keep a small set of rows for manual testing and combine the
    remaining records.
4.  Shuffle the data and remove the `title`, `subject`, and `date`
    columns.
5.  Clean the news text:
    -   Convert text to lowercase.
    -   Remove URLs, HTML tags, punctuation, newlines, and words
        containing digits.
    -   Remove bracketed text and other non-word characters.
6.  Split the data into training and testing sets (75% / 25%).
7.  Convert text into TF-IDF features using `TfidfVectorizer`.
8.  Train and evaluate four models:
    -   Logistic Regression
    -   Decision Tree Classifier
    -   Gradient Boosting Classifier
    -   Random Forest Classifier
9.  Enter a news article manually and view predictions from all four
    models.

## Project structure

``` text
Fake-News-Prediction/
│
├── FakeNewsPrediction(1).ipynb
├── Fake.csv
├── True.csv
├── README.md
└── requirements.txt
```

Place the two dataset CSV files in the same directory as the notebook
before running it.

## Requirements

-   Python 3.8 or later
-   Jupyter Notebook or JupyterLab
-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn

## Installation

Open a terminal in the project directory and install the dependencies:

``` bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

``` bash
jupyter notebook
```

Open `FakeNewsPrediction(1).ipynb` and run the cells from top to bottom.

## Dataset

The notebook expects two CSV files:

-   `Fake.csv`: news records labelled as fake (`class = 0`).
-   `True.csv`: news records labelled as true/not fake (`class = 1`).

The notebook uses the `text` column as the model input. Make sure both
files contain a `text` column and are available at the paths used in the
notebook.

## Models and evaluation

The notebook trains the following classifiers using the same TF-IDF
training features:

  Model                 Purpose
  --------------------- -----------------------------------------------
  Logistic Regression   Linear classification model
  Decision Tree         Rule-based tree classifier
  Gradient Boosting     Ensemble model that builds trees sequentially
  Random Forest         Ensemble of decision trees

Model performance is checked using the test-set score and classification
reports. Actual accuracy and precision/recall values depend on the
dataset and the train/test split; run the notebook to obtain your
results.

## Manual prediction

At the end of the notebook, the `manual_testing(news)` function accepts
a news text string and prints predictions from all four trained models.

Example input:

``` python
news = "Paste the news article text here."
manual_testing(news)
```

The function returns model predictions, not independent verification of
the article. Treat the output as an estimate and verify important news
with reliable sources.

## Notes

-   The notebook uses `train_test_split` without a fixed `random_state`,
    so results may vary between runs.
-   The notebook's Decision Tree classification-report cell currently
    passes `pred_lr` instead of `pred_dt`. For the correct Decision Tree
    report, change that cell to:

``` python
print(classification_report(y_test, pred_dt))
```

-   The dataset files are required to run the notebook but were not
    included with the uploaded notebook.
-   This is an educational machine learning project. It should not be
    used as the sole basis for deciding whether real-world news is true
    or false.

## Author

Sujit Karmakar, BCA(H), Techno India University, West Bengal.
