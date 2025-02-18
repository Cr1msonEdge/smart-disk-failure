## Task Description
Taking into account the monitoring data of the disk status [S.M.A.R.T](https://ru.wikipedia.org/wiki/S.M.A.R.T) and fault data, come up with your solution and determine if each disk will fail within the next 30 days.

## Dataset description
The dataset is presented as a `csv` file containing SMART data records of various disks over time periods. 
Attributes:
- `date` - date of recording
- `failure` - disk failure
- `model` - the name of the model
- `serial_number` - the serial number of the disk
- `smart 5` - the number of detected read/write errors
- `smart 9` - working hours
- `smart 187` - the number of errors that the drive reported to the host (computer interface) during any operations
- `smart 188` - the number of interrupted operations due to HDD timeout
- `smart 192` - the number of shutdown cycles or emergency failures
- `smart 197` - the number of sectors that are candidates for replacement.
- `smart 198` - the number of sectors that cannot be adjusted (by disk means). (critical defects)
- `smart 199` - the number of errors that occur when transmitting data over the interface
- `smart 240` - total time spent by the head unit in the working position in hours
- `smart 241` - the total number of recorded sectors.
- `smart 242` - the total number of sectors read.

Note, that the unlike the others tasks of smart HDD failure, the dataset in this task doesn't include smart normalized data (usually named as `smart_n_normalized`).

## Data preprocessing
### Clearing data 
As a result of a exploratory analysis of the data, the following was found out:
- the `model` attribute can be deleted because there is only one model.
- the `capacity_bytes` attribute can be deleted for the same reason.
- the `smart_raw_198` attribute is equal to the `smart_raw_197` attribute, so it can also be deleted (this was noticed by the correlation matrix, and it is also indicated in the [documentation seagate](https://t1.daumcdn.net/brunch/service/user/axm/file/zRYOdwPu3OMoKYmBOby1fEEQEbU.pdf#page=7.10))
- no `nan` values found

### Normalization
After observing EDA, we may see that the data is strongly assymetrical. We tried to normalize it.
The following data normalization methods have been tested:
- logarithmization
- the box-cox transformation
- mixed logarithmization and box-cox transformations (selectively for different columns)
- **Yeo-Johnson transformation** 
- normalization using formulas from [documentation seagate](https://t1.daumcdn.net/brunch/service/user/axm/file/zRYOdwPu3OMoKYmBOby1fEEQEbU.pdf#page=7.10)

### Feature engineering
Due to the time component in the task, it was decided to add the following features:
- shifted signs (the value of the same sign, but for the previous day) (`shift_smart_n_raw_`)
- the difference of signs (the difference between the current value of the sign and the shifted one) (`diff_smart_n_raw_`)

## Solution methods
The following solution methods were used
- Gradient boosting models (XGBoost, LightGBM, CatBoost)
- Two-layer XGBoost (for more information, see the documentation)
- Weighted ensemble
- Blending
- Stacking
## Results
The results of the best experiments.

For **class 0**

| Metric name / <br>model name| accuracy  | precision     | recall        | f1            | ROC-AUC       |
| ---------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- |
| CatBoost                                 | **0.999** | 0.966         | 0.554         | 0.704         | 0.777         |
| LightGBM                                 | **0.999** | 0.791         | 0.490         | 0.605         | 0.745         |
| XGBoost                                  | **0.999** | 0.940         | 0.677         | **0.787** II  | 0.838         |
| Double Layered XGBoost                      | **0.999** | 0.808         | **0.696** III | 0.748         | **0.848** III |
| Weighted ensemble                                 | **0.999** | **0.969** III | 0.624         | 0.761         | 0.813         |
| Blending                                 | **0.999** | **0.986** I   | 0.564         | 0.717         | 0.782         |
| Blending (SMOTE)                         | **0.999** | 0.915         | **0.733** II  | **0.814** I   | **0.866** II  |
| Stacking (meta_model: CatBoost)           | **0.999** | 0.519         | **0.789** I   | 0.627         | **0.894** I   |
| Stacking (meta_model: LogisticRegression) | **0.999** | **0.979** II  | 0.642         | **0.775** III | 0.821         |

For **class 0** metrics precision, recall, f1 are equal 1
