import streamlit as st 
import pandas as pd
import numpy as np
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
#import statsmodels.formula.api as smf
#import statsmodels.stats.api as sms
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from scipy import stats, linalg

import streamlit as st  
import os
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
st.set_page_config(page_icon=":house:", page_title="Rent Prediction")

st.write("Streamlit version", st.__version__)
st.set_option('deprecation.showPyplotGlobalUse', False)

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

#df= df[(df["lat"]< 55) & (df['lat'] > 20)]

#df= df[(df["long"]< -20) & (df['long'] > -110)] 
 
def main_page():    
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('images/Christmas2.webp') 
    
    st.markdown("# Welcome to the Rent Predictor 🏘🎄")
    st.markdown("### In the sidebar to the left there are several pages that can take you through the machine learning side of the predictor.")
    st.markdown("### If you wish to go straight to the predictor select that page.")
    from PIL import Image 
    image1 = Image.open('images/house3.jpeg')
    st.image(image1)

    st.markdown("# ENJOY!")
    st.sidebar.markdown("# Welcome!🎈❄️")
 
def page2():
    st.markdown("# Importing Libraries and Loading Data ")
   
    
    code = '''#Importing libraries
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
    import statsmodels.api as sm
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.linear_model import LinearRegression
    
    #Read in the dataset into a dataframe
    df = pd.read_csv("USAHousing2.csv")
    
    import random
    random.seed(42)
    df = df.sample(frac = 0.80)
    df.shape
    
    '''
    st.code(code, language= 'python')
    
    df = pd.read_csv("USAHousing2.csv")
    
    import random
    random.seed(42)
    df = df.sample(frac = 0.80)
    st.write('The dataset has the shape: ')
    
    
    if st.checkbox("Show number of rows and columns"):
        st.write(f'Rows: {df.shape[0]}')
        st.write(f'Columns: {df.shape[1]}')
    
    
    
    
    
    
    from PIL import Image 
    #st.sidebar.markdown("# Loading Data")
    image28 = Image.open('images/load.png')
    st.sidebar.image(image28)

def page3():
    st.markdown("# Exploratory Data Analysis") 
    
    
    from PIL import Image 
    #st.sidebar.markdown("# Loading Data")
    image29 = Image.open('images/eda.webp')
    st.sidebar.image(image29)
    
    df = pd.read_csv("USAHousing2.csv")
    
    import random
    random.seed(42)
    df = df.sample(frac = 0.80)
    
    tab, tab2, tab3 = st.tabs([" Univariate Analysis ", " Variable Distribution/Rent Option Frequencies "," Variable Plots "])
     
    with tab: 
        df = pd.read_csv("USAHousing2.csv")
        st.markdown("### Explore the dataset")
        #Display the first 10 rows of the data
        df = pd.read_csv("USAHousing2.csv")
        
        data = st.selectbox('Select Dataset: ', ['USAHousing2.csv'])
        if  data == 'USAHousing2.csv':
                st.markdown("#### Enter number of rows to explore:")
                rows = st.number_input("", min_value = 1, value = 5)
                if rows > 0:
                    st.dataframe(df.head(rows))
        
        st.markdown("##### Explore the tail end of the dataset")
        #st.experimental_show(df.tail())
        tailrows = st.number_input(" ", min_value = 1, value = 5)
        if tailrows > 0:
            st.dataframe(df.tail(tailrows))
        st.markdown("##### Descriptive Statistics")
        # Show dataset description
        if st.checkbox("Show description of dataset"):
            st.write(df.describe())
        from PIL import Image 
        
        st.markdown("##### Would you like to see the number of unique elements for each variable? ")
        unique = st.radio("   ", ('No', 'Yes'))
        if unique == 'Yes':
            st.write('You selected Yes.')
            st.experimental_show(df.nunique())
        else:
            st.write("You selected No.")
        
        st.markdown("#### Show Dataframe Information:")
        info = st.radio("    ", ('No', 'Yes'))
        if info == 'Yes':
            st.write('You selected Yes.')
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        else:
            st.write("You selected No.")
       

    with tab2:
    
       st.markdown("### Examine Variable Distributions:")
       df.hist(color = "green")
       plt.show()
       st.pyplot()
       
       st.markdown("### Select a Barplot to Display Rent Option Frequencies:")
       barplots = st.selectbox("           ", ['Parking', 'Laundry', 'Regions', 'State', 'Type'])
       if barplots == 'Parking':
           st.markdown("### Count of Parking Options:")
           st.bar_chart(df.parking_options.value_counts())
       elif barplots == 'Laundry':
           st.markdown("### Count of Laundry Options:")
           st.bar_chart(df.laundry_options.value_counts())
       elif barplots == 'Regions':
           st.markdown("### Top Five Most Common Regions:")
           st.bar_chart(df.region.value_counts().head())
       elif barplots == 'State':
           st.markdown("### Renting Options by State:")
           st.bar_chart(df.state.value_counts())
       elif barplots == 'Type':
           st.markdown("### Count of Different Renting Options:")
           st.bar_chart(df.type.value_counts().head(3))
        
       st.markdown("### Code for Narrowing Renting Options:")
    
       code = '''# Narrowing Renting Options to Apartments, Houses, and Townhouses
df = df[(df.type != 'condo') & (df.type!= 'duplex') & (df.type != 'manufactured')
&(df.type!= 'cottage/cabin') & (df.type != 'loft') & (df.type!= 'flat') 
& (df.type!= 'in-law') & (df.type!= 'land') &(df.type!= 'assisted living')]
    
        '''
       st.code(code, language= 'python')
    
    
    with tab3:
        st.markdown("### Dogs Allowed")
        st.bar_chart(df.dogs_allowed.value_counts())
        
        st.markdown("### Cats Allowed")
        st.bar_chart(df.cats_allowed.value_counts())
        
        st.markdown("### Electric Vehicle Charge")
        st.bar_chart(df.electric_vehicle_charge.value_counts())
        
        df=df[df["price"] > 200 ]
        df=df[df["sqfeet"]>= 200]

        df=df[df["price"]<2000]

        df= df[(df["sqfeet"]<= 1600) & (df['sqfeet'] > 300)]

        df=df[df["beds"]<= 3]

        df=df[df["baths"]<= 3.5]
        sns.set(font_scale=.6)
        
        st.markdown("##### To visually explore more features press the button:")
        if st.button("Display Boxplots"):
            st.write(sns.boxplot(x = 'beds', y = 'price', data = df))
            st.pyplot()
            st.write(sns.boxplot(x = 'laundry_options', y = 'price', data = df))
            st.pyplot()
            st.write(sns.boxplot(x = 'parking_options', y = 'price', data = df))
            st.pyplot()
            st.write(sns.boxplot(x = 'baths', y = 'price', data = df))
            st.pyplot()
        
        
        
    
def page4():
    st.markdown("# Handling Missing Data and Outliers")
    
    code = '''
    #Missing Values
df.isna().sum()
        '''
    st.code(code, language= 'python')
    
    if st.checkbox("Show Missing Values"):
        from PIL import Image 
        image9 = Image.open('missingvalues.png')
        st.image(image9)
    
    code = '''
    #Fill missing values with the mode
df["parking_options"] = df["parking_options"].fillna(df["parking_options"].mode()[0])
df["laundry_options"] = df["laundry_options"].fillna(df["laundry_options"].mode()[0])
df.fillna(0, inplace=True)

#Check Missing Values
df.isna().sum()
        '''
    st.code(code, language= 'python')
    from PIL import Image 
    image10 = Image.open('missingh.png')
    st.image(image10)
    
    st.markdown("## Finding and Handling Outliers")
    
    from PIL import Image 
    #st.sidebar.markdown("# Loading Data")
    image30 = Image.open('missingimage.png')
    st.sidebar.image(image30)
    
    
    df = pd.read_csv("USAHousing2.csv")
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("### Code for Boxplots of Outliers")    
    code = '''   
st.write(sns.boxplot(x = 'price', data = df))
st.pyplot()
   
st.write(sns.boxplot(x = 'sqfeet', data = df))
st.pyplot()
   
st.write(sns.boxplot(x = 'beds', data = df))
st.pyplot()
   
st.write(sns.boxplot(x = 'baths', data = df))
st.pyplot()
         '''
    st.code(code, language= 'python')
    
    st.write(sns.boxplot(x = 'price', data = df))
    st.pyplot()
    
    st.write(sns.boxplot(x = 'sqfeet', data = df))
    st.pyplot()
    
    st.write(sns.boxplot(x = 'beds', data = df))
    st.pyplot()
    
    st.write(sns.boxplot(x = 'baths', data = df))
    st.pyplot()
    
    st.markdown("## Fix the Data")
    if st.checkbox("Check to fix the data."):
        df=df[df["price"] > 200 ]
        df=df[df["sqfeet"]>= 200]
        df=df[df["price"]<2000]
        st.write(sns.boxplot(x = 'price', data = df))
        st.pyplot()
        df= df[(df["sqfeet"]<= 1600) & (df['sqfeet'] > 300)]
        st.write(sns.boxplot(x = 'sqfeet', data = df))
        st.pyplot()
        df = df[df["beds"]<= 3]
        st.write(sns.boxplot(x = 'beds', data = df))
        st.pyplot()
        df = df[df["baths"]<= 3.5]
        st.write(sns.boxplot(x = 'baths', data = df))
        st.pyplot()
     
    st.markdown("### Code for Outlier Trimming")    
    code = '''
     #Fill missing values with the mode
df=df[df["price"] > 200 ]
df=df[df["sqfeet"]>= 200]
df=df[df["price"]<2000]
st.write(sns.boxplot(x = 'price', data = df))
st.pyplot()
df= df[(df["sqfeet"]<= 1600) & (df['sqfeet'] > 300)]
st.write(sns.boxplot(x = 'sqfeet', data = df))
st.pyplot()
df = df[df["beds"]<= 3]
st.write(sns.boxplot(x = 'beds', data = df))
st.pyplot()
df = df[df["baths"]<= 3.5]
st.write(sns.boxplot(x = 'baths', data = df))
st.pyplot()
         '''
    st.code(code, language= 'python')
    
def page5():
    st.markdown("# Encoding/Correlation Exploration")
    
    st.markdown("## Label Encoding ")
    code = ''' 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["state"]=le.fit_transform(df["state"])
df["region"]=le.fit_transform(df["region"])
df["laundry_options"]=le.fit_transform(df["laundry_options"])
df["parking_options"]=le.fit_transform(df["parking_options"])
df["type"]=le.fit_transform(df["type"])
         '''
    st.code(code, language= 'python')
    
    st.markdown("## Correlation Barplot ")
    code = '''
import plotly.express as px
correlation = df.corr()["price"].reset_index().sort_values("prices", ascending = False)
fig = px.bar(correlation, x = "index", y = "price")
fig.show()

             '''
    st.code(code, language= 'python')
    
    from PIL import Image 
    #st.sidebar.markdown("# Loading Data")
    image31= Image.open('correlation.png')
    st.sidebar.image(image31)
    
    from PIL import Image 
    image19= Image.open('encode.png')
    st.image(image19)
    st.markdown("## Feature Correlation with Target ")
    code = ''' 
hm = df.corr()
plt.figure(figsize = (17, 15))
sns.heatmap(hm, annot = True, square = True, cmap = 'flare')
         '''
    st.code(code, language= 'python')
    
    
    image20= Image.open('heatmap.png')
    st.image(image20)
    
    code = '''
corr: pd.DataFrame = df.corr(method = "pearson")
corr['price'].sort_values(ascending = False)

             '''
    st.code(code, language= 'python')
    
    image21= Image.open('corr.png')
    st.image(image21)
    
    
    st.markdown("## Most Correlated Features")
    code = '''
sns.regplot(data = df, x="sqfeet", y= "price",
            line_kws{"color": "red"})

             '''
    st.code(code, language= 'python')
    image22= Image.open('regplots.png')
    st.image(image22)
    code = '''
sns.regplot(data = df, x="baths", y= "price",
            line_kws{"color": "red"})

             '''
    st.code(code, language= 'python')
    image50= Image.open('regbath.png')
    st.image(image50)
    code = '''
sns.regplot(data = df, x="laundry_options", y= "price",
            line_kws{"color": "red"})

             '''
    st.code(code, language= 'python')
    image51= Image.open('regl.png')
    st.image(image51)

def page6():
      st.markdown("# Model Creation")
      
      from PIL import Image 
      #st.sidebar.markdown("# Loading Data")
      image33 = Image.open('ML.jpeg')
      st.sidebar.image(image33)
      
      tab, tab2, tab3 = st.tabs(["Model Creation Code", "Validation Code","Results"])
      
      with tab:
          st.markdown("### Combine bed and bath so that these variables can be included in model.")
          code = '''
          df['bed_bath_total'] = df.apply(lambda x: x['beds'] + x['baths'], axis = 1)
            '''
          st.code(code, language= 'python')
          st.markdown("### Define Independent and Dependent Variables and Split the data")
          code = '''
X = df.drop(["price", 'long', 'lat','sqfeet', 'cats_allowed', 'beds', 'baths', ], axis=1) #independent variables
y = df["price"] #target variable
y = np.log(y)
#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
          '''
          st.code(code, language = 'python')
          st.markdown("### Create Two Pipelines to Test Two Models")
          code = '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor \n
p1 = Pipeline([('robust', RobustScaler()),
         ('linear', LinearRegression())])
p2 = Pipeline([('robust', RobustScaler()),
         ('random', RandomForestRegressor())]) \n
#Fit the pipeline on the training dataset
p1.fit(X_train, y_train) \n
#Fit the second pipeline on the training dataset
p2.fit(X_train, y_train) \n
#Make predictions
pred = p1.predict(X_train)
pred1 = p1.predict(X_test) \n
#Make predictions
pred2 = p2.predict(X_train)
pred22 = p2.predict(X_test) \n
#Evaluate model performance
print("R-Squared: ", round(r2_score(y_train, pred),2))     \n
#Evaluate model performance
print("R-Squared: ", round(r2_score(y_train, pred2),2))    
      
from statsmodels.stats.outliers_influence import variance_inflation_factor
      
# VIF dataframe
vifDF = pd.DataFrame()
vifDF["feature"] = X.columns
      
# calculating VIF for each feature
vifDF["VIF"] = [variance_inflation_factor(X.values, i)
    for i in range(len(X.columns))]
vifDF
          
          
          '''
          st.code(code, language = 'python')
          
      with tab2:
          st.markdown("## Validation Code")
          
          code = '''
#Root Mean Squared Error for Train
rmse1 = np.sqrt(mean_squared_error(y_train, p2.predict(X_train)))
rmse1
          
#Root Mean Squared Error for Test
rmse2 = np.sqrt(mean_squared_error(y_test, p2.predict(X_test)))
rmse2

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(p2, X_train, y_train, scoring='r2', cv=folds)
print("R2 scores: ", np.round(scores, decimals = 2))

print("Mean 5-Fold R2 score: ", round(scores.mean(),2))

#Looking at true values vs predicted values
pred_graph_ran=pd.DataFrame({"True Value":y_train, "Predicted Value":pred2})
pred_graph_ran.head()

#Evaluation Metrics
mae = metrics.mean_absolute_error(y_train, pred2)
mse = metrics.mean_squared_error(y_train, pred2)
rmse = np.sqrt(mse)

#View results of metrics
print("Result metrics:")
print("MAE:",mae)
print("RMSE:", rmse)
print("R2:", r2_score(y_train, pred2))
          
          '''
          st.code(code, language = 'python')
          
      with tab3:
          from PIL import Image 
          st.markdown("### R-Squared Value for Both Pipelines")
          code = '''
# Evaluate Model Performance
print("R-Squared: ", round(r2_score(y_train, pred), 2))
'''
          st.code(code, language = 'python')
          st.write(" R-Squared: 0.43")

          code = '''
# Evaluate Model Performance for second model
print("R-Squared: ", round(r2_score(y_train, pred), 2))
'''
          st.code(code, language = 'python')
          st.write(" R-Squared: 0.68")
          st.markdown("### Result Metrics")          
          code = '''
mae = metrics.mean_absolute_error(y_train, pred2)
mse = metrics.mean_squared_error(y_train, pred2)
rmse = np.sqrt(mse)

print("Result metrics:")
print("MAE:",mae)
print("RMSE:", rmse)
print("R2:", r2_score(y_train, pred2))
'''
          st.code(code, language = 'python')
          st.write("Result metrics: ")
          st.write("MAE: 0.12798362401968671")
          st.write("RMSE: 0.18242400886845558")
          st.write("R2: 0.6821160961378734 ")       
          
          st.markdown("### K-Fold Cross-Validation")
          
          code = '''
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(p2, X_train, y_train, scoring='r2', cv=folds)
print("R2 scores: ", np.round(scores, decimals = 2))
'''
          st.code(code, language = 'python')
          st.write("R-Squared scores: [0.61   0.62  0.60  0.61  0.61]")
          
          code = '''
print("Mean 5-Fold R2 score: ", round(scores.mean(),2))
'''
          st.code(code, language = 'python')
          st.write("Mean 5-Fold R-Squared score: 0.61")
          
          
          
          st.markdown("### First Five Predictions - Actual vs Prediction (Log)")
          code = '''
pred_graph_ran=pd.DataFrame({"True Value":y_train, "Predicted Value":pred2})
pred_graph_ran.head()
'''
          st.code(code, language = 'python')
          image26= Image.open('pred.png')
          st.image(image26)

          st.markdown("### Specific Prediction With Input") 
          st.write("Result converted back to original price scale")
          image27= Image.open('spred.png')
          st.image(image27)
          code = '''
pp = p2.predict([[0,1,1,0,0,0,4,4,6,4]])
print("Predicted rent price: ", np.exp(pp))
'''
          st.code(code, language = 'python')
          st.write("Predicted Rent Price: [1286]")
          
          st.markdown("### Test Multicollinearity") 
          code = '''
vif["VIF"] = [variance_inflation_factor(X.values, i)
              for i in range(len(X.columns))]
vif
'''
          st.code(code, language = 'python')
          image35= Image.open('vif.png')
          st.image(image35)
          
          
          st.markdown("### Confirm Assumptions")
          code = '''
testp = p2.predict(X_test)
resid = y_test - testp
plt.figure(figsize=(11,9))

sns.regplot(testp, y_test, scatter_kws ={'color':'b','alpha':0.1},color='g')
plt.ylabel('Y test')
plt.xlabel('Pred Y')
plt.title('Predicted VS. Test')
plt.show()
'''
          st.code(code, language = 'python')
           
          image36= Image.open('asump.png')
          st.image(image36)
          
          
          code = '''
plt.figure(figsize = (12,8))
stats.probplot(resid, dist = "norm", plot = plt)
plt.title("Normal Q-Q plot")
plt.show()
'''
          st.code(code, language = 'python')
          image37= Image.open('asump2.png')
          st.image(image37)
          
          code = '''
plt.figure(figsize=(11,8))
plt.scatter(testp ,resid , alpha = 0.2)
plt.xlabel('Predicted Home Price')
plt.ylabel('Residuals')
plt.title("Residual Plot")
plt.show()
'''
          st.code(code, language = 'python')
          
          image38= Image.open('asump3.png')
          st.image(image38)
          
          code = '''
#Calculate Durbin Watson
from statsmodels.stats.stattools import durbin_watson
np.round(durbin_watson(reg.resid), decimals = 2)
'''
          st.code(code, language = 'python')
          st.write("1.94")
          
def page7():
    
    st.markdown("# Welcome to the Rent Predictor🏠🎄❄️")
    
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('Christmas2.webp')    
    
    from PIL import Image 
    #st.sidebar.markdown("# Loading Data")
    image32 = Image.open('forrent.jpeg')
    st.sidebar.image(image32)
    
    st.header("Please select each of the following factors to discover the rent estimate:")

        
        
    
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
    
    
    htype = st.selectbox('Select Rent Housing Type ', ['Apartment', 'House', 'Townhouse'])
    if  htype == 'Apartment':
        htype = 0
    elif htype == 'House':
        htype = 1
    else:
        htype = 2
    
    dogs = st.selectbox('Dogs Allowed:', ['Yes', 'No'])
    if  dogs == 'Yes':
        dogs = 1
    else: dogs = 0
    
    smoking = st.selectbox('Smoking Allowed:', ['Yes', 'No'])
    if  smoking == 'No':
        smoking = 0
    else: smoking = 1
    
    wheelchair = st.selectbox('Wheelchair Access:', ['Yes', 'No'])
    if  wheelchair == 'No':
        wheelchair = 0
    else: wheelchair = 1
    
    electric = st.selectbox('Electric Vehicle Charge:', ['Yes', 'No'])
    if  electric == 'No':
        electric = 0
    else: electric = 1
    
    furnished = st.selectbox('Furnished:', ['Yes', 'No'])
    if  furnished == 'No':
        furnished = 0
    else: furnished = 1
        
    
    laundry = st.selectbox('Select Laundry Option:', ['In Unit', 'On Site', 'W/D Hookup', 'Laundry in Building', 'No Laundry on Site'])
    if  laundry == 'Laundry in Building':
        laundry  = 0
    elif laundry  == 'On Site':
        laundry  = 1
    elif laundry  == 'No Laundry on Site':
        laundry  = 2
    elif laundry == 'W/D Hookup':
        laundry  = 3
    elif laundry == 'In Unit':
        laundry = 4
    
    
    parking = st.selectbox('Select Parking Option:', ['Attached Garage',  'Carport','Detached Garage', 'Street Parking','Off-Street Parking', 'No Parking'])
    if  parking == 'Attached Garage':
        parking  = 0
    elif parking == 'Carport':
        parking = 1
    elif parking  == 'Detached Garage':
        parking = 2
    elif parking  == 'No Parking':
        parking  = 3
    elif parking  == 'Off-Street Parking':
        parking  = 4
    elif parking == 'Street Parking':
        parking  = 5
        
    state = st.selectbox('Select a State: ', ['Colorado','Florida', 'Georgia','Iowa','Louisiana','Michigan','North Carolina','New York', 'New Jersey', 'Pennsylvania', 'Tennessee', 'Texas','Virginia'])
    if  state == 'Texas':
        state = 40
    elif state == 'Florida':
        state = 9
    elif state == 'North Carolina':
        state = 25
    elif state == 'Colorado':
        state = 5
    elif state == 'New Jersey':
        state = 29
    elif state == 'New York':
         state = 31
    elif state == 'Pennsylvania':
        state = 35
    elif state == 'Louisiana':
        state = 16
    elif state == 'Michigan':
        state = 20
    elif state == 'Virginia':
        state = 42
    elif state == 'Georgia':
        state = 10
    elif state == 'Tennessee':
        state = 39
    elif state == 'Iowa':
        state = 11
    
    bedbath = st.number_input('Bedroom/Bathroom Total:', min_value=2, max_value=6, value=2, help = 'Add Bedrooms and then Bathrooms, EX: 2 = 1 bed 1 bath')
    
    if st.button('Predict Rent Price'):
        price = predict(htype, dogs, smoking, wheelchair, electric, furnished, bedbath, laundry, parking, state)
        st.write("Predicted Rent Price")
        st.success(np.exp(price).astype(int))
        
    
        
    st.markdown("## Happy Holidays! 🎄")

page_names_to_funcs = {
    "Welcome Page": main_page,
    "Loading Data": page2,
    "Exploratory Data Analysis": page3,
    "Missing Values/Outliers": page4,
    "Encoding/Correlation Exploration": page5,
    "Model Creation": page6,
    "Rent Predictor": page7
    
    }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

    
    
    
    
    
