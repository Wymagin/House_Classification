def solution(true_labels, predicted_labels):
    from sklearn.metrics import classification_report
    print(classification_report(true_labels, predicted_labels))
