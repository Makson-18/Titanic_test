import pandas as pd
import tensorflow as tf
from tensorflow import keras

url = "/content/train.csv" #тренировочные данные
url2 = "/content/test.csv" #тестовые данные

data_train = pd.read_csv(url) #создаём таблицу тренировчных данных
data_test = pd.read_csv(url2) #создаём таблицу тестовых данных

data_train["Pclass"] = (data_train["Pclass"] > 2).astype(int) #признаки класса пассажира. Если 3 то 1, иначе 0
data_train["Sex_parser"] = (data_train["Sex"] == "female").astype(int)  #признаки пола. Если женщина, то 1. Иначе 0
middle_age = data_train["Age"].mean() #среднее значения возраста
data_train["Young"] = (data_train["Age"] < middle_age).astype(int) #признаки возраста. Если меньше среднего то 1, иначе 0
data_train["Familly"] = data_train["SibSp"] + data_train["Parch"] + 1 #признаки родствеников. 1 для счета самого пассажира
data_train["Familly"] = (data_train["Familly"] < 2).astype(int) #признаки родствеников. Меньше 1, то 1. Иначе 0

clean_table = data_train.dropna(subset="Age") #Создаем таблицу, удаляя пропускии возраста
x = clean_table[["Pclass", "Sex_parser", "Young", "Familly"]] #Объединяем признаки в таблицу
y = clean_table[["Survived"]] #Признаки выживших

model = keras.Sequential([
    keras.layers.Dense(128, activation="relu"), #Начальный слой
    keras.layers.Dense(64, activation="relu"), #Скрытый слой
    keras.layers.Dense(1, activation="sigmoid") #Выходной слой
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) #Оптимизируем код, валидацию убрал для 100 процентов данных.
model.fit(x, y, epochs=50) #Обучаем

#делаем тоже самое для тестовых данных
data_test["Pclass"] = (data_test["Pclass"] > 2).astype(int)
data_test["Sex_parser"] = (data_test["Sex"] == "female").astype(int)
data_test["Age"] = data_test["Age"].fillna(0).astype(int) #заполняем пропуски нулями.
data_test["Young"] = (data_test["Age"] < middle_age).astype(int)
data_test["Familly"] = data_test["SibSp"] + data_test["Parch"]
data_test["Familly"] = (data_test["Familly"] < 2).astype(int)

x_test = data_test[["Pclass", "Sex_parser", "Young", "Familly"]] # объединяем признаки в таблицу
prediction = model.predict(x_test[:10]) #берем первые десять данных
prediction_digits = (prediction > 0.5).astype(int).flatten() #больше 0.5 то 1. Иначе 0. Записываем в строку для удобства
print(prediction_digits) #Выводим результат
