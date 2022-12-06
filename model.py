import pandas as pd
    
    import time 
    
    import numpy as np
    
    #Importing libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn import preprocessing
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.linear_model import LinearRegression
    
    #Read in the dataset into a dataframe
    df = pd.read_csv("USAHousing2.csv")
    
    import random
    random.seed(42)
    df = df.sample(frac = 0.80)
    
    df = df[(df.type != 'condo') & (df.type!= 'duplex') & (df.type != 'manufactured') 
            & (df.type!= 'cottage/cabin') & (df.type != 'loft') & (df.type!= 'flat') & (df.type!= 'in-law') &
            (df.type!= 'land') &(df.type!= 'assisted living')]
    
    #Missing values
    df.isna().sum()
    #Fill missing values with the mode
    df["parking_options"] = df["parking_options"].fillna(df["parking_options"].mode()[0])
    df["laundry_options"] = df["laundry_options"].fillna(df["laundry_options"].mode()[0])
    df.fillna(0, inplace=True)
    
    #Remove irrelevant features
    #df.drop(columns = ["id", "url", "region_url", "image_url", "description"],axis = 1,inplace = True)
    
    
    #Fix dataset so that it does not include Zero for price and sqfeet
    df=df[df["price"] > 200 ]
    df=df[df["sqfeet"]>= 200]
    
    df=df[df["price"]<2000]
    
    df= df[(df["sqfeet"]<= 1600) & (df['sqfeet'] > 300)]
    
    df=df[df["beds"]<= 3]
    
    df=df[df["baths"]<= 3.5]
    
    
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    df["state"]=le.fit_transform(df["state"])
    #df["region"]=le.fit_transform(df["region"])
    df["laundry_options"]=le.fit_transform(df["laundry_options"])
    df["parking_options"]=le.fit_transform(df["parking_options"])
    df["type"]=le.fit_transform(df["type"])
    
    #df.drop("region", axis=1, inplace=True)
    
    
    df['bed_bath_total'] = df.apply(lambda x: x['beds'] + x['baths'], axis = 1)
    
    X = df.drop(["price", 'sqfeet', 'cats_allowed', 'beds', 'baths', ], axis=1) #independent variables
    y = df["price"] #target variable
    y = np.log(y)
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    from sklearn.pipeline import Pipeline
    
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestRegressor
    
    p1 = Pipeline([('robust', RobustScaler()),
                   ('linear', LinearRegression())])
    
    p2 = Pipeline([('robust', RobustScaler()),
                   ('random', RandomForestRegressor())])
    
    #Fit the pipeline on the training dataset
    p1.fit(X_train, y_train)
    
    #Fit the pipeline on the training dataset
    p2.fit(X_train, y_train)
    
    
    #Make predictions
    pred = p1.predict(X_train)
    pred1 = p1.predict(X_test)
    
    
    #Make predictions
    pred2 = p2.predict(X_train) 
    pred22 = p2.predict(X_test)
    
    pp = p2.predict([[0,1,1,0,0,0,4,4,6,4]])
    #print("Predicted rent price: ", np.exp(pp)) 
    
    
    import pickle
    
    
    pickle.dump(p2, open('p2.pkl', 'wb'))
    
    pickled_model = pickle.load(open('p2.pkl', 'rb'))
    
    
    pickled_model.predict(X)
    
    def predict(htype, dogs, smoking, wheelchair, electric, furnished, bedbath, laundry, parking, state):
        prediction = p2.predict(pd.DataFrame([[htype, dogs, smoking, wheelchair, electric, furnished, bedbath, laundry, parking, state]], columns = ['htype', 'dogs', 'smoking', 'wheelchair', 'electric', 'furnished', 'bedbath', 'laundry', 'parking', 'state']))
        return prediction
    
    
