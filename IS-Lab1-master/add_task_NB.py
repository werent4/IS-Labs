'''
Implementation of Naive Bayes for 1st lab (Additional task)
'''

from sklearn.naive_bayes import GaussianNB
from typing import List, Tuple
from pathlib import Path

def create_dataset(data_path: Path) -> Tuple[List[List[float]], List[int]]:
    x1, x2, target = [], [], []
    with open(data_path) as file:
        for line in file:
            values = line.strip().split(',')
            x1.append(float(values[0]))
            x2.append(float(values[1]))
            target.append(float(values[2]))
    return [x1,x2], target

data_path = Path("Data.txt")
(x1, x2), target = create_dataset(data_path)

X = [[x1[i], x2[i]] for i in range(len(x1))]
y = target

test_size = int(0.1 * len(target))
X_train = X[:-test_size]
X_test = X[-test_size:]
    
y_train = y[:-test_size]
y_test = y[-test_size:]

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

print("Trying to predict:", )
y_pred = nb_classifier.predict(X_test)
print(y_pred)
print("correct answers:\n",y_test)

