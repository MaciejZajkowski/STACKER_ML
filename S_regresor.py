import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split
from random import randint

class Stacker_Regresion:
    """ 
        This model is Stacker of regresors, it trains provided models
        Then trains provided clasifier that will chouse what model to use during each prediction
        When using keras you need to compile model beafore you put in here
    """
    def __init__(self,models=[],Final_model = [],training_sets_split = 0.2,Boosting = False):
        Stacker_Regresion.models = models
        Stacker_Regresion.Final_model = Final_model
        Stacker_Regresion.training_sets_split = training_sets_split
        Stacker_Regresion.Trained_models = [False for i in range(len(models))]
        Stacker_Regresion.Boosting = Boosting
        
    def get_clousest_clases(self,X,Y):
        """
            Gets predictions for every model in models
            then chouses witch model is best for each prediction (mse) 
            returnes data frame of clases and dataframe of all predictions
        """
        
        err  =[]
        predictions = []
        cnt = 0
        for model in Stacker_Regresion.models:
            if Stacker_Regresion.Trained_models[cnt]:
                pred = model.predict(X)
                if(pred.shape[0] != Y.shape):
                    pred = pred.reshape(1,-1)
                predictions.append(pred.squeeze())
                err.append((np.square( pred**2 - Y **2)).squeeze())
            cnt +=1
        classes = []
        for i in range(len(err[0])):
            temp = []
            for j in range(len(err)):
                temp.append(err[j][i])
            classes.append(temp.index(min(temp)))
            
        return pd.DataFrame(classes,columns=['Best_model']), pd.DataFrame(predictions)
    
    def fit(self,X,Y,models_to_train = [],random_state = randint(1,100)):
        """
            separates data then fits models, and clasifier to chouse what data is the best 
            returns table of history (if you are using keras, in skl models returns just model names so use it wisely)
            
            models_to_train requaiers list of boolian type where place in list coresponds to each model, you need to pass true if you wan to train model
            training every model is deafoult, it fills extra empty spots by True, so if you want to train all models exept first you can do it 
            
            random state is used to train test split, it is important to use your own random state when training models separetly - if not it may cause mixing
            datasets for difrent lines of model - it will cause clasifier to malfunction -  
            
            
        """
        if len(models_to_train) != len(Stacker_Regresion.models) and len(models_to_train) < len(Stacker_Regresion.models):
            for i in range( len(Stacker_Regresion.models) -len( models_to_train) ):
                models_to_train.append(True)
        from IPython.display import clear_output
        " Line 1 of model -  "
        X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X,Y, test_size=Stacker_Regresion.training_sets_split,random_state=random_state)
        hist = []
        #fiting models
        cnt = 0
        for model in Stacker_Regresion.models:
            clear_output(wait=True)
            if models_to_train[cnt]:
                print("training model, name: {}".format(model))
                hist.append(model.fit(X_train1,Y_train1))
                Stacker_Regresion.Trained_models[cnt] = True
            cnt += 1
        
            
        classes,pred = Stacker_Regresion.get_clousest_clases(self,X_train2,Y_train2)
        
        if Stacker_Regresion.Boosting:
            F_set = pd.concat([pd.DataFrame( X_train2) ,pd.DataFrame(pred)],axis=1,join='inner').values
            hist.append( Stacker_Regresion.Final_model.fit(F_set,Y_train2))
        else:
            hist.append( Stacker_Regresion.Final_model.fit(X_train2,classes))
        
        return hist
        
    def predict(self,X):
        predictions = []
        cnt = 0
        for model in Stacker_Regresion.models:
            if Stacker_Regresion.Trained_models[cnt]:
                pred = model.predict(X)
                if(pred.shape[0] != X.shape[0]):
                    pred = pred.reshape(1,-1)
                predictions.append(pred.squeeze())
            cnt+=1
        
        if Stacker_Regresion.Boosting:
            F_set = pd.concat([pd.DataFrame( X) ,pd.DataFrame(predictions)],axis=1,join='inner').values.reshape(-1,1)
            final_pred =Stacker_Regresion.Final_model.predict(F_set)
        else:    
            temp = Stacker_Regresion.Final_model.predict(X)
            if(temp.shape[0] != X.shape[0]):
                    temp = temp.reshape(1,-1)
            temp = temp.squeeze()
            
            final_pred = []
            for i in range(len(temp)):
                final_pred.append(predictions[temp[i]][i])
            
        return np.array(final_pred),np.array(predictions)
    
    def evaluate(self,Y, pred):
        #TODO conditions for squese
        x= []
        x.append(skl.metrics.median_absolute_error(Y,pred))
        print("Median absolute error: " ,x[0])
        
        x.append(skl.metrics.mean_absolute_percentage_error(Y,pred))
        print("Median absolute % error: " ,x[1])
        
        x.append(skl.metrics.mean_squared_log_error(Y,pred))
        print("Mean sq log error: " ,x[2])
        
        x.append(skl.metrics.mean_squared_error(Y,pred))
        print("Mean sq error: " ,x[3])
        
        x.append(skl.metrics.r2_score(Y,pred))
        print("r2 score: " ,x[4])
        return x
    
    def evaluate_models(self,X,Y):
        """evaluating every trained model and final stacker
        """
        f_pred,pred = Stacker_Regresion.predict(self = self,X=X)
        print("\nEvaluating Final model: \n")
        Stacker_Regresion.evaluate(self=self,Y=Y,pred=f_pred)
        for i in range(len(Stacker_Regresion.models)):
            if Stacker_Regresion.Trained_models[i]:
                print("\nEvaluating model: {} \n".format(Stacker_Regresion.models[i]))
                Stacker_Regresion.evaluate(self = self,Y=Y,pred = pred[i])
        