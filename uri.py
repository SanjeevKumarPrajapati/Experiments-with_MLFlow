import mlflow
print("Printing Tracking URI")
print(mlflow.get_tracking_uri())

#how to set your own URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
#Printing New URI ID below

print(mlflow.get_tracking_uri())
