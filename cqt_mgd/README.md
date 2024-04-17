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

python -u model_main.py --eval --eval_output=gru_result95.txt --model_path=gru3_models/model_logical_spect_100_32/epoch_95.pth>gru_eval95.txt

python -u resnet_main.py --eval --eval_output=resnet_result95.txt --model_path=resnet/model_logical_mgd_100_32/epoch_95.pth>resnet_eval95.txt

python -u em_test5.py --eval --eval_output=s1s2_0.01_result1.txt --model_path=s1s2_0.01model/model_logical_spect_spect_100_64/epoch_100.pth>s1s2_0.01_eval1.txt


python -u em_test6.py --eval --eval_output=lr_s1s3_0.01_result3.txt --model_path=s1s3_0.01model/model_logical_spect_spect_100_64/epoch_98.pth>lr_s1s3_0.01_eval3.txt

python -u em_test4.py --eval --eval_output=s1s3_0.01_result1.txt --model_path=s1s3_0.01model/model_logical_spect_spect_100_64/epoch_100.pth>s1s3_0.01_eval1.txt



python -u unet_main.py --eval --eval_output=unet_result90.txt --model_path=unet_models/model_logical_spect_100_32_0.0001/epoch_90.pth>unet_eval90.txt

#----------------------------------------------------------------------------------


python -u unet_main.py --num_epochs=100  --track=logical --features=spect  >ta.txt

python -u scg_main.py --num_epochs=100 --track=logical --features=spect > cbam.txt
```
python -u scg_main.py --eval --eval_output=cbam_result98.txt --model_path=cbam_models/model_logical_spect_100_32_0.0001/epoch_98.pth>cbam_eval98.txt

python -u evaluate_tDCF_asvspoof19.py>s2s3_0.0001_tdcf3.txt
python -u s2s3_3.py --eval --eval_output=s2s3_0.00001_result4.txt --model_path=s2s3_0.00001model/model_logical_spect_spect_100_64/epoch_90.pth>s2s3_0.00001_eval4.txt


python -u s2s3.py --eval --eval_output=s2s3_0.001_result5.txt --model_path=s2s3_0.00001model/model_logical_spect_spect_100_64/epoch_100.pth>s2s3_0.001_eval5.txt

python -u s1s2.py --num_epochs=100 --track=logical --batch_size=64 --features1=spect --features2=spect >s1s2_0.00001.txt

python -u s2s3_3.py --num_epochs=100 --track=logical --batch_size=64 --features1=spect --features2=spect >s2s3_0.00001.txt


python -u s1s2.py --eval --eval_output=s1s2_emb_result.txt --model_path=model/s1s2_0.001_epoch_97.pth>s1s2_emb_eval.txt


python -u s1s3.py --eval --eval_output=s1s3_emb_result.txt --model_path=s1s3_0.001model/model_logical_spect_spect_100_64/epoch_98.pth>s1s3_emb_eval.txt


python -u em_test4.py --eval --eval_output=s1s3_0.01_result3.txt --model_path=s1s3_0.01model/model_logical_spect_spect_100_64/epoch_98.pth>s1s3_0.01_eval3.txt

python -u main_pick.py --num_epochs=5 --track=logical --features1=spect --features2=mgd >pick.txt
python -u pick_traindata.py >pick.txt

Then compute the evaluation scores using on the development dataset

```
python evaluate_tDCF_asvspoof19.py RESULTS_FILE PATH_TO__asv_dev.txt 
```




'''
python model_main.py --num_epochs=100 --track=[logical/physical] --features=[spect/mfcc/cqcc]   --lr=0.00005



python model_main.py --eval  --eval_output=RESULTS_FILE --model_path=CHECKPOINT_FILE
'''