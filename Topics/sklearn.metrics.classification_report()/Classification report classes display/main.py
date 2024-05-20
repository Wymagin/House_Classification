
def solution(true_labels, predicted_labels):
    from sklearn.metrics import classification_report
    print(classification_report(true_labels, predicted_labels, target_names=['ApPlE', 'BaNaNa', 'OrAnGe', 'PeAr']))