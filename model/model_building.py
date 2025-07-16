import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# normally this is not needed
# but the columns of this data are hard to interpret, 
# so we rename it to ease our understanding
def rename_columns(df):
    df_new = df.rename(
    columns={   # ----- numerical
                # 'age': 
                # 'sex':
                'trestbps':'blood-pressure', # resting blood pressure (on admimission to hospital), mm Hg
                'chol':'cholesterol', # serum cholestoral, mg/dl
                'thalch':'max-heart-rate', # max heart rate achieved
                'oldpeak':'st-depression', # ST (S wave, T wave) depression value induced by exercise relative to rest, numerical
                'ca':'num-vessel-affected', # number of major vessels (0-3) colored by flourscopy
                #-----categorical
                'dataset':'region', # dataset collected at 4 different regions/cities/countires
                'cp': 'chest-pain', # 4 types
                'fbs':'blood-sugar', # whether fasting blood sugar > 120 mg/dl
                'restecg':'ecg', # resting electrocardiogram
                'exang':'exe-angina', # exercise induced angina, True/False
                'slope': 'st-slope', # ST segment slope
                'thal':'mps-defect', # myocardial perfusion scan (blood flow to heart muscle) defect type
                #---- target
                'num':'heart-attack-num'
            })
    return df_new

df_original = pd.read_csv('./heart_disease_uci.csv')
"""label encoding for all categorical features (to reduce number of features) """
# rename columns
df = rename_columns(df_original)
df['heart-attack'] = np.where(df['heart-attack-num'].values>=1, 1, 0) # add another column to binarize the target

# zero cholesterol and boold-pressure are extreme outliers; replace with mean
feature_list = ['cholesterol', 'blood-pressure']
for fea in feature_list:
    df[fea][df[fea]==0] = df[fea].mean()

# Numerical features, replacing null with mean
feature_list = ['age', 'blood-pressure', 'cholesterol', 'max-heart-rate', 'st-depression', 'num-vessel-affected']
for fea in feature_list:
    df[fea] = df[fea].fillna(df[fea].mean())

# categorical features, label encoding
# true/false/nan -> 1/0/2
feature_list = ['sex', 'region', 'blood-sugar','exe-angina'] + ['st-slope','chest-pain', 'ecg']
label_encoder = LabelEncoder()
for fea in feature_list:
    df[fea] = label_encoder.fit_transform(df[fea]) 

# feature selection 
df = df.drop(columns=['id']) # unique ID, useless for modeling
df = df.drop(columns=['mps-defect','num-vessel-affected']) # >50% missing
df = df.drop(columns=['cholesterol']) # drop this, as it does not show much correlation with target
df = df.drop(columns=['region']) # drop this, as its correlation with target doesn't make sense

# train/test split
X = df.iloc[:,:-2]
y = df.iloc[:,-1] # use binary target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

steps = [
        #  ('scalar', StandardScaler()),
        #  ('pca', PCA(n_components=15)),
         ('xgb', XGBClassifier(
        n_estimators = 200,
        subsample = 0.5, # subsample ratio of the training set
        max_depth=10, 
        min_child_weight = 2, # min sum of instance weight needed in a child
        gamma = 0.1, # min loss reduction needed to make further partition
        reg_alpha = 10, # L1
        reg_lambda= 10,  # L2
        colsample_bytree = 0.6, # subsample ratio of columns when constructing a tree
        colsample_bylevel = 0.6, # subsample ratio of columns when constructing a level    
        random_state=42,                      
        )),
        ]
model = Pipeline(steps)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
