
**[Master branch](https://github.com/Victor4869/open-alcnet)**

Fix reported issues in the orginal scripts.

**[Dev branch](https://github.com/Victor4869/open-alcnet/tree/dev)**

This branch contains additional features and improvements:
1. The [SIRST](https://github.com/YimianDai/sirst) and [IRSTD-1k](https://github.com/RuiZhang97/ISNet) datasets added to the folder.
2. Added argument to pass the path of the ALCNet folder in Colab.
3. Added argument to pass the root path of the dataset if you are using a dataset outside the ALCNet folder.
4. Improved logging messages
5. Improved file management:

>* Relocate the training log file and parameter file to \params folder. 
>* Relocate the log file and prediction result to \results foler.

6. Addtional features in the visualize_local_contrast_networks.py scrpit:

>* Record the inference time.
>* Draw the ground truth boudning box in the prediction images for easier comparsion, this feature requires the labelling (.xml) file in the dataset.

7. Addtional features in the train_alcnet.py:
>* Record and plot the training losses and validation losses.
>* Monitor the losses, create model checkpoints when signs of overfitting are detected.
>* Record and plot the IoU and nIoU.

8. Included some sample commands for running the scripts.

***
If you are unfamiliar with how to setup an environment for MXNet or how to run the scripts, you can check out the tutorial in the [Wiki](https://github.com/Victor4869/open-alcnet/wiki).
***
# Description from the orginal author

codes and trained models for the ALCNet

## Requirements
 
Install [MXNet](https://mxnet.apache.org/) and [Gluon-CV](https://gluon-cv.mxnet.io/):

```
pip install --upgrade mxnet-cu100 gluoncv
```

## Dataset

The SIRST dataset: <https://github.com/YimianDai/sirst>

## Experiments 

The trained model params are in `./params`

## Citation

Please cite our paper in your publications if our work helps your research. BibTeX reference is as follows.

```
@inproceedings{dai21acm,
  title   =  {Asymmetric Contextual Modulation for Infrared Small Target Detection},
  author  =  {Yimian Dai and Yiquan Wu and Fei Zhou and Kobus Barnard},
  booktitle =  {{IEEE} Winter Conference on Applications of Computer Vision, {WACV} 2021}
  year    =  {2021}
}

@article{TGRS21ALCNet,
  author    = {{Dai}, Yimian and {Wu}, Yiquan and {Zhou}, Fei and {Barnard}, Kobus},
  title   = {{Attentional Local Contrast Networks for Infrared Small Target Detection}},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  pages   = {1--12},
  year    = {2021},
}
```
