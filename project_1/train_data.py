import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import argparse


def load_data(path):
    # load labeled data
    df = pd.read_csv(path)
    return df


def train(df:pd.DataFrame):
    # set Feature and Target
    X_text = df['question'].tolist()
    y = df['label_int'].astype(int)

    # embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_embed = model.encode(X_text)

    # split dataset
    X_train, _, y_train, _ = train_test_split(
        X_embed, y, test_size=0.2, random_state=42, stratify=y
    )

    # training logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # export model
    joblib.dump(clf, 'resources/model/trained_model.pkl')


if __name__ == "__main__":
    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='resources/data/labeled_question_list_all.csv', help='Path to file CSV data')
    args = parser.parse_args()

    df = load_data(args.data)
    train(df)

    print('Model berhasil ditraining!')
