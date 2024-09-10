import json
from sklearn.metrics import classification_report

def predict(groundtruth_file, prediction_file):

    labels = ['negative', 'neutral', 'positive', 'unrelated']

    total_y_test = []
    total_y_pred = []

    with open(f"{groundtruth_file}", 'r') as infile:
        lines = infile.readlines()
    
    y_test = [json.loads(line.strip())["annotation"] for line in lines]

    with open(f"{prediction_file}", 'r') as infile:
        predict_lines = infile.readlines()
    
    y_pred = [json.loads(line.strip())["label"] for line in predict_lines]

    total_y_test.extend(y_test)
    total_y_pred.extend(y_pred)

    print(classification_report(y_test, y_pred, target_names=labels))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth-file", type=str, default=None)
    parser.add_argument("--prediction-file", type=str, default=None)
    args = parser.parse_args()

    predict(args.groundtruth_file, args.prediction_file)
    