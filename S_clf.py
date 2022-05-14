from operator import contains
import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Stacker_Classifier:
    """ 
        This model is Stacker of regresors, it trains provided models
        Then trains provided clasifier that will chouse what model to use during each prediction
        When using keras you need to compile model beafore you put in here
    """
    def __init__(self,models=[],Final_model = [],training_sets_split = 0.2,Boosting = False):
        Stacker_Classifier.models = models
        Stacker_Classifier.Final_model = Final_model
        Stacker_Classifier.training_sets_split = training_sets_split
        Stacker_Classifier.Trained_models = [False for i in range(len(models))]
        Stacker_Classifier.Boosting = Boosting
        
    
    def fit(self,X,Y,models_to_train = [],random_state = randint(1,100)):
        """
            separates data then fits models, and clasifier to chouse what data is the best 
            returns table of history (if you are using keras, in skl models returns just model names so use it wisely)
            
            models_to_train requaiers list of boolian type where place in list coresponds to each model, you need to pass true if you wan to train model
            training every model is deafoult, it fills extra empty spots by True, so if you want to train all models exept first you can do it 
            
            random state is used to train test split, it is important to use your own random state when training models separetly - if not it may cause mixing
            datasets for difrent lines of model - it will cause clasifier to malfunction -  
            
            m            
            
        """
        if len(models_to_train) != len(Stacker_Classifier.models) and len(models_to_train) < len(Stacker_Classifier.models):
            for i in range( len(Stacker_Classifier.models) -len( models_to_train) ):
                models_to_train.append(True)
        from IPython.display import clear_output
        " Line 1 of model -  "
        X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X,Y, test_size=Stacker_Classifier.training_sets_split,random_state=random_state)
        hist = []
        #fiting models
        cnt = 0
        for model in Stacker_Classifier.models:
            clear_output(wait=True)
            if models_to_train[cnt]:
                print("training model, name: {}".format(model))
                hist.append(model.fit(X_train1,Y_train1))
                Stacker_Classifier.Trained_models[cnt] = True
            cnt += 1
        
        predictions =[]
        cnt = 0
        for model in Stacker_Classifier.models:
            if Stacker_Classifier.Trained_models[cnt]:
                pred = model.predict(X)
                if(pred.shape[0] != X.shape[0]):
                    pred = pred.reshape(1,-1)
                predictions.append(pred.squeeze()) 
            cnt += 1
         
        if Stacker_Classifier.Boosting:
            F_set = pd.concat([pd.DataFrame( X_train2) ,pd.DataFrame(np.array(predictions).reshape(-1,1))],axis=1,join='inner')
            hist.append( Stacker_Classifier.Final_model.fit(F_set,Y_train2))
        else:
            hist.append( Stacker_Classifier.Final_model.fit(X_train2,Y_train2))
        
        return hist
        
    def predict(self,X):
        predictions = []
        cnt = 0
        for model in Stacker_Classifier.models:
            if Stacker_Classifier.Trained_models[cnt]:
                pred = model.predict(X)
                if(pred.shape[0] != X.shape[0]):
                    pred = pred.reshape(1,-1)
                predictions.append(pred.squeeze())
            cnt+=1
        
        if Stacker_Classifier.Boosting:
            F_set = pd.concat([pd.DataFrame(X) ,pd.DataFrame(np.array(predictions).reshape(-1,1))],axis=1,join='inner')
            final_pred =Stacker_Classifier.Final_model.predict(F_set)
            
        else:    
            temp = Stacker_Classifier.Final_model.predict(X)
            if(temp.shape[0] != X.shape[0]):
                    temp = temp.reshape(1,-1)
            temp = temp.squeeze()
            
            final_pred = []
            for i in range(len(temp)):
                final_pred.append(predictions[temp[i]][i])
            
        return np.array(final_pred),np.array(predictions)
    
    def evaluate(self,Y, pred):
        #TODO conditions for squese
        cm =confusion_matrix(Y,pred)
        sns.heatmap(cm)
        
        
    def evaluate_models(self,X,Y):
        """evaluating every trained model and final stacker
        """
        f_pred,pred = Stacker_Classifier.predict(self = self,X=X)
        print("\nEvaluating Final model: \n")
        cnt =0
        Stacker_Classifier.evaluate(self=self,Y=Y,pred=f_pred)
        for i in range(len(Stacker_Classifier.models)):
            if Stacker_Classifier.Trained_models[i]:
                print("\nEvaluating model: {} \n".format(Stacker_Classifier.models[i]))
                Stacker_Classifier.evaluate(self = self,Y=Y,pred = pred[i-cnt])
            else:
                cnt+=1
        