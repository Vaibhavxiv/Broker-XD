import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

#Pipeline Function

def build_pipeline(num_attribs,cat_attribs):
        num_pipeline=Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ])

        #pipeline for categorical values
        cat_pipeline=Pipeline([
            ('onehot',OneHotEncoder(handle_unknown="ignore"))
        ])

        #full pipeline
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])

        return full_pipeline

if not os.path.exists(MODEL_FILE):
        #Training the model
        data=pd.read_csv("housing.csv")

        # creating stratisfied test and train set
        data['income_cat']=pd.cut(data['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
        split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
        for train_range , test_range in split.split(data,data['income_cat']):
          housing=data.loc[train_range].drop('income_cat',axis=1)
          data.loc[test_range].drop('income_cat',axis=1).to_csv("input.csv",index=False)

        housing_labels=housing['median_house_value'].copy()
        housing_features=housing.drop('median_house_value',axis=1)

        num_attribs=housing_features.drop('ocean_proximity',axis=1).columns.tolist()
        cat_attribs=['ocean_proximity']

        pipeline=build_pipeline(num_attribs,cat_attribs)
        housing_prepared = pipeline.fit_transform(housing_features)

        model=RandomForestRegressor(random_state=42)
        model.fit(housing_prepared,housing_labels)

        joblib.dump(model,MODEL_FILE)
        joblib.dump(pipeline,PIPELINE_FILE)

        print("model is trained.")

else:
      #Loading the saved model & pipeline
      model=joblib.load(MODEL_FILE)
      pipeline=joblib.load(PIPELINE_FILE)

      input_data=pd.read_csv("input.csv")
      transformed_input=pipeline.transform(input_data)
      predictions=model.predict(transformed_input)
      input_data["median_house_value"]=predictions

      input_data.to_csv("output.csv")
      print("Results saved to output.csv")