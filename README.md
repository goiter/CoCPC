# Copula-based Contrastive Prediction Coding (Co-CPC)

This repository releases the code and data for stock movement prediction by considering the coupling with macroeconomic indicators. 
Please cite the follow paper if you use this code.

G. Wang, L. Cao, et.al, Coupling Macro-Sector-Micro Financial Indicators for LearningStock Representations with Less Uncertainty, AAAI 2021.

Should you have any query please contact me at [wgf1109@gmail.com](mailto:wgf1109@gmail.com).

## Dependencies

- Python 3.6.9
- Pytorch 1.0.0
- Scipy 1.3.1

## Directories

- data: dataset consisting of ACL18 dataset, KDD17 dataset and some macroeconomic variables in varied time intervals.
- log: store trained model for prediction. Our trained model for acl18 dataset (our_model-v1-model_best.pth) and kdd17 dataset (our_model-v_kdd17-model_best.pth) are provided.


## Data Preprocess

Due to the upload limit, we just upload the smaller one preprocessed data in pickle format. For larger data, you can generate the data by running load_data.py file.
- ACL18 ([Stocknet Data](https://github.com/yumoxu/stocknet-dataset)): similar to the original paper, we generate each batch data with size 32. The preprocess operation
 can be refer to file load_data.py, here we provide the preprocessed file in pickle format. 
- KDD17: The raw data is from [Adv-ALSTM](https://github.com/fulifeng/Adv-ALSTM/tree/master/data/kdd17/price_long_50), we process them in load_data.py file, here we also provide the prepprocessed file in pickle format.
- Macroeconomic indicators: The data is from [FRED](https://fred.stlouisfed.org/).


## Running

- directly use pre-trained model for prediction:
 
    python main.py --version v1  --epochs 30

 
- train Co-CPC model:
    
    python main.py --cpc_train True --version v1 
    


