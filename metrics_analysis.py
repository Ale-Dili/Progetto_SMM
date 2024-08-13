import pickle

def best_parameter(metrics):
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    for combo,values in metrics.items():
        if values['accuracy'] > best_accuracy:
            best_accuracy = values['accuracy']
            best_combo_accuracy = combo
        if values['precision'] > best_precision:
            best_precision = values['precision']
            best_combo_precision = combo
        if values['recall'] > best_recall:
            best_recall = values['recall']
            best_combo_recall = combo
        if values['f1'] > best_f1:
            best_f1 = values['f1']
            best_combo_f1 = combo
    return best_combo_accuracy, best_combo_precision, best_combo_recall, best_combo_f1, best_accuracy, best_precision, best_recall, best_f1

with open('metrics.pkl', 'rb') as f:  # open a text file
    metrics = pickle.load(f) # serialize the list

best_combo_accuracy, best_combo_precision, best_combo_recall, best_combo_f1, best_accuracy, best_precision, best_recall, best_f1 = best_parameter(metrics)
#print the best parameters for each metric
print(f'Best parameters for accuracy: {best_combo_accuracy} with accuracy: {best_accuracy}')
print(f'Best parameters for precision: {best_combo_precision} with precision: {best_precision}')
print(f'Best parameters for recall: {best_combo_recall} with recall: {best_recall}')
print(f'Best parameters for f1: {best_combo_f1} with f1: {best_f1}')