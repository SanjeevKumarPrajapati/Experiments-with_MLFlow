import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Setting the URI 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

#Load wine dataset
df=load_wine()
x=pd.DataFrame(df.data, columns=df.feature_names)
x = x.astype('float64')  # optional: avoid int-related schema issues
y=df.target


#train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=42)

#Define the params of RF
max_d=5
n_es=8

mlflow.set_experiment("MLFlow_Experiment-1")

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_d,n_estimators=n_es,random_state=42)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_d)
    mlflow.log_param('n_estimators',n_es)
    
    #creating confusion matrix
    a=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(4,5))
    sns.heatmap(a,annot=True,fmt='d',cmap='Blues',xticklabels=df.target_names,yticklabels=df.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    
    #save plot
    plt.savefig("confusion_matrix.png")
    
    #log artifact using mlflow
    mlflow.log_artifact("confusion_matrix.png")
    
    #set tags
    mlflow.set_tags({"Author":"Sanjeev","Project":"Wine Classification"})
    
    # Create input example
    input_example = pd.DataFrame(X_train[:2], columns=df.feature_names)
    
    #log the model
    mlflow.sklearn.log_model(sk_model=rf, name="RandomForestClassifier",input_example=input_example,registered_model_name="RandomForestClassifier")
    
    mlflow.log_artifact(__file__)
    print(accuracy)
