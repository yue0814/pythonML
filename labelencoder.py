from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df[].values)
class_le.inverse_transform(y)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_feature=[0])
# return a sparse matrix so use toarray()
ohe.fit_transform(X).toarray()

# only convert string columns
pd.get_dummies

