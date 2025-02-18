## EDA

The most important results from the preliminary analysis were as follows:
- removal of uninformative features
    - `capacity_bytes`, `model`, `serial_number`, `failure` - uninformativity
    - `smart_198_raw` - the same as `smart_197_raw`
- No `nan` values found
- significant class imbalance

#### Implementation
Implementation in `eda.py`. The file contains the construction of various graphs, time series for certain discs (i.e. discs with certain serial numbers), the construction of histograms, boxes with whiskers, etc.


## Data processing
### Class imbalance
To solve the problem of class imbalance, class weights for models (gradient boosting) were specified, as well as SMOTE generation of synthetic data of various types, Adasyn, and undersampling methods. 
**The most important thing to note** is that due to the strong imbalance, synthetic data generation methods **greatly reduce** the value of the **precision** metric, but slightly increase recall.
### Normalization
The following data normalization methods have been tested:
- logarithmization
- Box-Cox transformations
- mixed logarithmization and Box-Cox transformations (optional for different columns)
- **Yeo-Johnson transformations**
- normalization according to formulas from [documentation seagate](https://t1.daumcdn.net/brunch/service/user/axm/file/zRYOdwPu3OMoKYmBOby1fEEQEbU.pdf#page=7.10) (`smart_n_normalized`)

### Getting new features
Due to the time component in the task, it was decided to add the following features:
- shifted signs (the value of the same sign, but for the previous day) (`shift_smart_n_raw_n)
- feature difference (the difference between the current feature value and the shifted one) (`diff_smart_n_raw_n`)
- normalized by formulas from [documentation seagate](https://t1.daumcdn.net/brunch/service/user/axm/file/zRYOdwPu3OMoKYmBOby1fEEQEbU.pdf#page=7.10 ) signs (`smart_n_normalized`)
#### Implementation
Implementation in `preprocessing.py `. The classes presented in this file implement various normalization methods, dataset partitioning, as well as a method for adding a target variable, since it is initially missing from the dataset; data type conversion to reduce memory consumption; and removal of low-importance features. By changing the order in which operations are applied, we change the result. If you first split and then normalize, then normalization will be performed for each resulting part separately.

As a result of the transformations, Yeo-Johnson were most often used, since they support negative values that could be obtained as a result of creating shifted features and difference ones; and transformations using seagate normalization formulas were also often used. 

![[./images/Pasted image 20241223002553.png]]
*feature importance*


## Solution methods
### Gradient boosting models
Different gradient boosting models were tested: XGBoost, CatBoost, and LightGBM.
Hyperparameter optimization was implemented for these classes using various methods: GridSearch, RandomSearch, and TPE optimization.
Implementation in `MyModel.py `

### The ensemble
An ensemble of gradient boosting methods was tested: XGBoost, CatBoost, and LightGBM. The algorithm's prediction is counted as a weighted sum of the three models.  
Implementation in `XCL.py `. It is possible to optimize the hyperparameters of the three models, as well as optimize the weights of the models. But overall it's better for the models to have equal coeffiient weights.
##### Realization
Implementation in `XCL.py `.
### Two-layer XGBoost
The basic idea is that the first tuning of the first XGBoost model takes place, remembering the important signs. Next, it learns from scratch on stratified K fold (but with optimized parameters) and makes predictions on validation samples that we memorize (similar to stacking). The second model is trained on the important features of the initial dataset + the predictions of the first model. That is, the final prediction is made by the model of the second layer, using as a dataset the combination of important features of the initial dataset and the predictions of the model of the first layer.
##### Realization
Implementation in `DoubleLayer.py `. Before that, LightGBM was used as the main model due to its fast operation. It is possible to use ready-made model settings.
### Blending
##### Realization
Implementation in `Blending.py `. MLP, naive Bayes classifier, CatBoost, XGBoost, Random Forest, and Logistic Regression were tested as models. In the end, the combination that showed better results was random forest, XGBoost, CatBoost, and CatBoost, or Logistic Regression as a meta-model. CatBoost was chosen because it shows good results out of the box, i.e. without hyperparameter optimization, Random Forest and XGBoost were used with optimized parameters.
### Stacking
##### Realization
Implementation in `StackingSK.py `. The models selected are all the same as in the blending. The best set: random forest, CatBoost, XGBoost and meta-models: Logistic Regression
## Logging
After completing each iteration of training and testing (preprocessing, training, parameter optimization, testing), comprehensive information about the iteration, namely
* Models used (for algorithms using multiple models)
* Preprocessor operations
* Optimization operations
* Metrics
saved in .save the json file to the saved folder. A dump of the model(s), as well as converted datasets, is also saved. 
*The repository does not store logs of all tests, we have left only the most important ones. We also didn't upload model and dataset dumps.*