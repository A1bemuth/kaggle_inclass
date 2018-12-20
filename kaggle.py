#!/usr/bin/env python
# coding: utf-8

# ## Пример решения

# Kaggle

# In[1]:


import numpy
import pandas
import sklearn

import xgboost as xgb


# Загружаем датасет

# In[2]:


train = pandas.read_csv('Train.csv')
test = pandas.read_csv('Test.csv')


# In[3]:


print(train.shape)


# In[4]:


train.head()


# С помощью метода describe() получим некоторую сводную информацию по всей таблице. По умолчанию будет выдана информация только для количественных признаков. Это общее их количество (count), среднее значение (mean), стандартное отклонение (std), минимальное (min), макcимальное (max) значения, медиана (50%) и значения нижнего (25%) и верхнего (75%) квартилей:

# In[5]:


train.describe()


# Узнать количество заполненных (непропущенных) элементов можно с помощью метода count. Параметр axis = 0 указывает, что мы двигаемся по размерности 0 (сверху вниз), а не размерности 1 (слева направо), т.е. нас интересует количество заполненных элементов в каждом столбце, а не строке:

# In[6]:


train.count(axis=0)


# Заполнить пропущенные значения можно с помощью метода fillna. Заполним, например, медианными значениями.
# 
# axis=0 по-прежнему указывает, что мы двигаемся сверху вниз:

# In[7]:


train = train.fillna(train.median(axis=0), axis=0)


# In[8]:


train.count(axis=0)


# In[9]:


train['date'].describe()


# In[10]:


categorical_columns = [c for c in train.columns if train[c].dtype.name == 'object']
numerical_columns   = [c for c in train.columns if train[c].dtype.name != 'object']
print(categorical_columns)
print(numerical_columns)


# пока что уберем колонку дата из выборки

# Подготавляваем признаки

# In[11]:


y_train = train['price'].values


# In[12]:


X_train = train[['street_id',
          'build_tech',
          'floor',
          'area',
          'rooms',
          'balcon',
          'metro_dist',
          'g_lift',
          'n_photos',
          'kw1',
          'kw2',
          'kw3',
          'kw4',
          'kw5',
          'kw6',
          'kw7',
          'kw8',
          'kw9',
          'kw10',
          'kw11',
          'kw12',
          'kw13']].values


# In[13]:


X_test = test[['street_id',
          'build_tech',
          'floor',
          'area',
          'rooms',
          'balcon',
          'metro_dist',
          'g_lift',
          'n_photos',
          'kw1',
          'kw2',
          'kw3',
          'kw4',
          'kw5',
          'kw6',
          'kw7',
          'kw8',
          'kw9',
          'kw10',
          'kw11',
          'kw12',
          'kw13']].values


# Выполняем преобразование признаков

# In[14]:


sklearn.preprocessing.MinMaxScaler().fit_transform(X_train, y_train)


# Обучаться, или, как говорят, строить модель, мы будем на обучающей выборке, а проверять качество построенной модели – на тестовой. Разбиение на тестовую и обучающую выборку должно быть случайным. Обычно используют разбиения в пропорции 50%:50%, 60%:40%, 75%:25% и т.д.
# 
# Мы воспользуемся функцией train_test_split из модуля sklearn.cross_validation. и разобьем данные на обучающую/тестовую выборки в отношении 70%:30%:

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 
print(N_train, N_test)


# GBT – градиентный бустинг деревьев решений¶
# GBT – еще один метод, строящий ансамбль деревьев решений. На каждой итерации строится новый классификатор, аппроксимирующий значение градиента функции потерь.

# In[18]:


from sklearn import ensemble


# In[ ]:


gbt = ensemble.GradientBoostingClassifier(n_estimators=50, verbose=2)
gbt.fit(X_train, y_train)


# In[ ]:


err_train = np.mean(y_train != gbt.predict(X_train))
print(err_train)


# In[ ]:


print(gbt.predict(X_test))

