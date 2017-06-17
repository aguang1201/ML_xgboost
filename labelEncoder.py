from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
encoder.fit(["paris", "paris", "tokyo", "amsterdam"])
