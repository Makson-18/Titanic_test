# импортируем нужные библиотеки 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('/content/train.csv')# загружаем тренировочные данные
data_test = pd.read_csv('/content/test.csv')# загружаем тестовые данные

# высчитываем медиану, т.е. число по середине
median_age = data_train['Age'].median()
median_fare = data_train['Fare'].median()

def data_table(data):
  data['Sex_Parser'] = (data['Sex'] == 'female').astype(int)#  #признаки пола. Если женщина, то 1. Иначе 0
  data['Pclass'] = (data['Pclass'] > 2).astype(int)# признаки класса пассажира. Если 3 то 1, иначе 0
  data['Young'] = (data['Age'].fillna(median_age) < 30).astype(int)# заполняем медиану для пропусков возраста
  data["Large_Family"] = (((data['SibSp'] + data['Parch']) + 1) > 2).astype(int)# #признаки родствеников. Больше 2, то 1. Иначе 0
  data['Fare_Final'] = data['Fare'].fillna(median_fare)# заполняем медиану для пропусков цены билета

  features  = ['Sex_Parser', 'Pclass', 'Young', 'Large_Family', 'Fare_Final']# все признаки
  return data[features]

x = data_table(data_train)# таблица признаков
y = data_train['Survived']# число выживших

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)# валидация 20%

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)# древо для табличных данных
model.fit(x_train, y_train)# обучение модели

train_preds = model.predict(x_train)# предсказания для тренировочных данных
val_preds = model.predict(x_val)# предсказания для валидационных данных

print(f"Accuracy на обучении: {accuracy_score(y_train, train_preds) * 100:.2f}%")# выводим тренировочные данные
print(f"Accuracy на валидации: {accuracy_score(y_val, val_preds) * 100:.2f}%")# выводим валидационные данные

x_test = data_table(data_test)# тестовые данные
prediction = model.predict(x_test)# предсказания для тестовых данных
print(f"Выжившие на первых 10 данных: {prediction[:10]}")# выводим предсказания для первых 10 пассажиров

# Сохраняем результат в формате CSV для загрузки на Kaggle
submission = pd.DataFrame({
    'PassengerId': data_test['PassengerId'],
    'Survived': prediction
})

submission.to_csv("submission.csv", index=False)
print("Файл submission.csv успешно создан!")
