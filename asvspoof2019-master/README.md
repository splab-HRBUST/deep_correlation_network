# Deep Residual Neural Networks for Audio Spoofing Detection

This repo has an implementation for our paper **Deep Residual Neural Networks for Audio Spoofing Detection**, this is also describes the solution of team **UCLANESL** in the [ASVSpoof 2019 competition](https://www.asvspoof.org/).

## Dataset

The ASVSpoof2019 dataset can be downloaded from the following link:

[ASVSpoof2019 dataset](https://datashare.is.ed.ac.uk/handle/10283/3336)

### Training models
```
python model_main.py --num_epochs=100 --track=[logical/physical] --features=[spect/mfcc/cqcc]   --lr=0.00005
```

Please note that the CQCC features are computing using the Matlab code in [cqcc_extraction.m](./cqcc_extraction.m), so you need to run this file to generate cache files of CQCC featurs before attempting to traiin or evaluate models with CQCC features.

#### To perform fusion of multiple results files
```
 python fuse_result.py --input FILE1 FILE2 FILE3 --output=RESULTS_FILE
```

### Evaluating Models

Run the model on the evaluation dataset to generate a prediction file.
```
python model_main.py --eval  --eval_output=RESULTS_FILE --model_path=CHECKPOINT_FILE

python -u model_main.py --eval  --eval_output=scores.txt --model_path=resnet/models/model_logical_spect_100_32/epoch_99.pth> test_eval.txt

python -u model_main.py --eval  --eval_output=scores.txt --model_path=models/model_logical_spect_200_32_5e-05/epoch_199.pth > test_eval.txt

python -u scg_main.py --num_epochs=100 --track=logical --features=spect > scg.txt
```

resnet/models/model_logical_spect_100_32/epoch_99.pth




Then compute the evaluation scores using on the development dataset

```
python evaluate_tDCF_asvspoof19.py RESULTS_FILE PATH_TO__asv_dev.txt 
```




'''
python model_main.py --num_epochs=100 --track=[logical/physical] --features=[spect/mfcc/cqcc]   --lr=0.00005



python model_main.py --eval  --eval_output=RESULTS_FILE --model_path=models/model_logical_spect_200_32_5e-05/epoch_199.pth
'''