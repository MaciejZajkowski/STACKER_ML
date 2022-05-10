import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split

class Stacker_Regresion:
    """ 
        This model is Stacker of regresors, it trains provided models
        Then trains provided clasifier that will chouse what model to use during each prediction
        When using keras you need to compile model beafore you put in here
    """
    def __init__(self,models=[],Classifier = [],training_sets_split = 0.2):
        Stacker_Regresion.models = models
        Stacker_Regresion.Classfier = Classifier
        Stacker_Regresion.training_sets_split = training_sets_split
        
    def get_clousest_clases(self,X,Y):
        """
            Gets predictions for every model in models
            then chouses witch model is best for each prediction (absolute error) 
            returnes data frame of clases and dataframe of all predictions
        """
        
        err  =[]
        predictions = []
        for model in Stacker_Regresion.models:
            pred = model.predict(X)
            if(pred.shape[0] != Y.shape):
                pred = pred.reshape(1,-1)
            predictions.append(pred.squeeze())
            err.append((pred - Y).squeeze())
            
        classes = []
        for i in range(len(err[0])):
            temp = []
            for j in range(len(err)):
                temp.append(err[j][i])
            classes.append(temp.index(min(temp)))
            
        return pd.DataFrame(classes,columns=['Best_model']), pd.DataFrame(predictions)
    
    def fit(self,X,Y):
        """
            separates data then fits models, and clasifier to chouse what data is the best 
            returns table of history (if you are using keras, in skl models returns just model names so use it wisely)
            
        """
        
        from IPython.display import clear_output
        " Line 1 of model -  "
        X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X,Y, test_size=Stacker_Regresion.training_sets_split)
        
        hist = []
        for model in Stacker_Regresion.models:
            clear_output(wait=True)
            print("training model, name: {}".format(model))
            hist.append(model.fit(X_train1,Y_train1))
        
        classes,pred = Stacker_Regresion.get_clousest_clases(self,X_train2,Y_train2)
        
        hist.append( Stacker_Regresion.Classfier.fit(X_train2,classes))
        
        return hist
        
    def predict(self,X):
        predictions = []
        
        for model in Stacker_Regresion.models:
            pred = model.predict(X)
            if(pred.shape[0] != X.shape[0]):
                pred = pred.reshape(1,-1)
            predictions.append(pred.squeeze())
            
            
        temp = Stacker_Regresion.Classfier.predict(X)
        if(temp.shape[0] != X.shape[0]):
                temp = temp.reshape(1,-1)
        predictions.append(temp.squeeze())
        
        final_pred = np.array([])
        for i in range(len(predictions[0])):
            np.append(final_pred,(predictions[ predictions[-1][i] ][i]))
            
        return np.array(final_pred),np.array(predictions[:-1])
    
    def evaluate(self,y, pred):
        #TODO conditions for squese
        x= []
        x.append(skl.metrics.median_absolute_error(y,pred))
        print("Median absolute error: " ,x[0])
        
        x.append(skl.metrics.mean_absolute_percentage_error(y,pred))
        print("Median absolute % error: " ,x[1])
        
        x.append(skl.metrics.mean_squared_log_error(y,pred))
        print("Mean sq log error: " ,x[2])
        
        x.append(skl.metrics.mean_squared_error(y,pred))
        print("Mean sq error: " ,x[3])
        
        x.append(skl.metrics.r2_score(y,pred))
        print("r2 score: " ,x[4])
        return x
    
    def evaluate_models(self,X,Y):
        
        f_pred,pred = Stacker_Regresion.predict(self = self,X=X)
        print("\nEvaluating Final model: \n")
        Stacker_Regresion.evaluate(Y,f_pred)
        for i in range(len(pred)):
            print("\nEvaluating model: {} \n".format(Stacker_Regresion.models[i]))
            Stacker_Regresion.evaluate(Y,pred[i])
            