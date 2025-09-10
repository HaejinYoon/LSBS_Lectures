import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
print(df.head(2))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='target'), 
    df['target'], 
    test_size=0.3, 
    random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000, random_state = 0)
model.fit(X_train, y_train)

np.set_printoptions(suppress=True)
y_prob_org = model.predict_proba(X_test)
model.predict(X_test)
y_prob_org[:,1] # 암 환자라고 확신하는 정도
(y_prob_org[:,1] > 0.5).astype(int)

print(pd.DataFrame(y_prob_org[:].round(3)))

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
isp = ConfusionMatrixDisplay(confusion_matrix=cm)
isp.plot(cmap=plt.cm.Blues)
plt.show()
y_test[:9]
a = np.array(1, 0, 0, 1, 1, 0, 0, 0, 1)
b = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1])

cm = confusion_matrix(y_test[:9], b)
isp = ConfusionMatrixDisplay(confusion_matrix=cm)
isp.plot(cmap=plt.cm.Blues)
plt.show()

bb = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1])
cm = confusion_matrix(y_test[:9], bb)
isp = ConfusionMatrixDisplay(confusion_matrix=cm)
isp.plot(cmap=plt.cm.Blues)
plt.show()




#==================================================================================================================
# 다중 분류 지표 이해하기
#==================================================================================================================
import pandas as pd 
import numpy as np 
from sklearn import set_config
set_config(display="diagram")
#pd.set_option('display.max_columns', None) # 모든 칼럼이 출력되게 조절

from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="micro")
f1_macro = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision_macro:.2f}")
print(f"Recall: {recall_macro:.2f}")
print(f"F1 Score: {f1_macro:.2f}")