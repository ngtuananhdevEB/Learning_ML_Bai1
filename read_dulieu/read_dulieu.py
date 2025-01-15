import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaler
import seaborn as sn
from seaborn import scatterplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
data = pd.read_csv("diabetes.csv")

#In các kiểu dữ liệu của cột
# print(data.info())

#được sử dụng để tính ma trận hệ số tương quan giữa các cột
#result = data.corr()

#số lượng giá trị duy nhất trong cột "Outcome"
# print(data["Outcome"].value_counts())

#dezine một bieu do
# plt.figure(figsize=(8,8))
# sn.histplot(data["Outcome"])
# plt.title("Diabetes")
# plt.savefig("Diabetes.png")

#b1: Phan chia du lieu theo chieu doc: feauter and targe
x= data.drop("Outcome", axis = 1)
y = data["Outcome"]
#b2: Phan chia du lieu 3 phan: train - val - test
#train_test_split
# Chia dữ liệu thành train và test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Tien xu ly du lieu
# z = (x-u)/s x:g tri ban dau, u: ki vong (mean_), s; do lech chuan
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# result = scaler.fit_transform(x_train[["Pregnancies"]])

#fit_transform: kết hợp giua fit và transform
#fit: đo ti le
#transform: may biến đổi theo ti le, khi da fit
# print(scaler.mean_)
# print(sqrt(scaler.var_))
# print()
# for i,j in zip(x_train[["Pregnancies"]].values, result):
#     print("Before {} After {}".format(i,j))

#Mô hình dự đoán dữ liệu
cls = SVC()
cls.fit(x_train,y_train)
#-------------------------
#test
y_predict = cls.predict(x_test)
# for i,j in zip(y_test, y_predict):
#     print("Actual {} Predicted {}".format(i,j))
#thong ke ti le du doan
#print(classification_report(y_test, y_predict))

#Bieu do thong ke gia tri thuc te - gia tri du doan
cm = np.array(confusion_matrix(y_test, y_predict, labels=[0,1]))
confusion = pd.DataFrame(cm, index=["Not Diabetic","Diabetic"], columns=["Not Diabetic","Diabetic"])
sn.heatmap(confusion, annot=True, fmt="g")
plt.savefig("diabetes_predictions")

# Tối đa hoá việc sử dụng dữ liệu
#K fold cross validation cv








