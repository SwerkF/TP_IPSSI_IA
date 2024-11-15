from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    predictions = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, predictions)
    return accuracy, auc
