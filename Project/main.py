import pandas as pd
x=pd.read_csv("C:/Users/3step for lp top/Desktop/data analitycs/tcc_ceds_music.csv")#music data set


# data=x.describe()#task3
# print(data)




# task4
# y = x.drop_duplicates()
# y = x.dropna()
# print (y)




#task5.1
import matplotlib.pyplot as plt
# plt.hist(x.len, color = "red")
# plt.title("Len graph")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()


#task5.2
# plt.boxplot(x.len, notch=True, vert=False)
# IQR = x.len.describe()['75%']-x.len.describe()['25%']
# print("IQR=", IQR)
# plt.show()



#task 6
#Histogram +ve right skew
#outlier > Q3 + 1.5 * IQR
#outlier < Q1 - 1.5 * IQR



#task7
#DBScan

# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# x = pd.read_csv("C:/Users/3step for lp top/Desktop/data analitycs/tcc_ceds_music.csv")
# data=x[['len', 'music']]
# dbscan = DBSCAN(eps= 0.28, min_samples = 20)
# pred = dbscan.fit(data)
# plt.scatter(data["len"], data["music"])
# outliers = data[pred.labels_ == -1]
# plt.scatter(outliers["len"], outliers["music"], color = 'r')
# plt.show()





# # #task8.1
# #decison tree
# import numpy as np
# from sklearn import tree
# x1 = x[['len']]
# y1 = x['music']
# model = tree.DecisionTreeRegressor()
# model.fit(x1, y1)
# predictions = model.predict(x1)
# print(model.feature_importances_)
# for index in range(len(predictions)):
#   print('Actual: ', y1[index], 'Prtdicttd :	', predictions[index])




# ##task8.2
#KNN
# import pandas as pd
# import numpy as np
# from sklearn import neighbors
#
# x2 = x[['len']].values
# y2 = x['music']
#
# model = neighbors.KNeighborsRegressor(n_neighbors = 5)
# model.fit(x2, y2)
# predictions = model.predict(x2)
#
# for index in range(len(predictions)):
#   print('Actual: ', y2[index], 'Prtdicttd :	', predictions[index], 'Weight', x2[index,0])

















##task 9 text mining
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import string

import pandas as pd
x=pd.read_csv("C:/Users/3step for lp top/Desktop/data analitycs/tcc_ceds_music.csv")#music data set
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

with open("C:/Users/3step for lp top/Desktop/data analitycs/Dataset.txt") as f:
  lines = f.readlines()
for i in lines:
  result1 = i.translate(str.maketrans('', '', string.punctuation))
  #print(result1)
  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(result1)
  result2 = [i for i in tokens if not i in stop_words]
  #print(result2)
  stemmer = PorterStemmer()
  for word in result2:
    stem = stemmer.stem(word)
    #print(stem)
  lemmatizer = WordNetLemmatizer()
  for word in tokens:
    lem = lemmatizer.lemmatize(word)
    #print(lem)cores(line))