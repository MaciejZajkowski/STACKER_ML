Stacker Regressor / Classifier

Stacker is a concept of ML model that has two layers of ML models. First layer consists of various Classifiers / Regressors and Learns how to predict on given data. Second layer consists of single classifier that learns form data witch model defined in previous layer ls best for current prediction. Model has option “Boosting = True” witch changes pipeline, then classifier input is not only  data but all predictions of stackers together with data. Model is compatible with scickit-learn API, and partially with tensorflow keras (final classifier is not compatible).  
Below we can see training process:


![obraz](https://user-images.githubusercontent.com/101389064/173014204-1be6614e-e70a-4823-9474-9aefc46533bc.png)


Pipeline of predict method consists of classifier chousing what prediction use (not finished – for now all models are predicting for all data, than only output is selected based on classifier selections).


Prediction Process wiwhaout Boosting:
![obraz](https://user-images.githubusercontent.com/101389064/173148186-07eeaed0-3623-4cc0-a428-9d2026835bca.png)


Prediction process with Boosting:

![obraz](https://user-images.githubusercontent.com/101389064/173150394-739d256a-2713-4147-b083-8f7602d5d99f.png)
 (picture might be bit confusing but classifier predicts witch prediction to use)


As test file proves that this model for now is not exactly best one – because it trains every model on same data, so classifiers / regressors are not specialized (only specialization is due to theirs different architecture – witch may be a lot in only some cases).  We can see that final classifier adds its own error to final predictions. 

TODO: Implementing smart data split - (every model in first layer is trained on different data, or at least partially (idea of making some buffer zone).  General idea is to split data to different models by different cases (maybe using k - means algorithm) – main goal is to make each model an expert in some use case.  

Possible use case: data with a lot of missing values, each classifier/ regressor will train on different type of data. 
