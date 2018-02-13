# Depth Estimation by Convolutional Neural Networks

You can read the whole thesis 
<a href="http://www.fit.vutbr.cz/study/DP/DP.php?id=18852&file=t"> here</a>. 

## Architecture:

Architecture similar to the one used by <a href="http://www.cs.nyu.edu/~deigen/depth/">Eigen <i>et al.</i></a> with the difference.

## Trained model

You can download the trained model <a href="https://www.dropbox.com/s/rki8o74r7yv0k8d/model_norm_abs_100k.caffemodel?dl=0">here</a>.

## Results:

All experiments were performed  on <a href="http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html">NYU Depth v2</a> dataset.

## Usage

`python test_depth.py INPUT_DIR GT_DIR OUT_DIR SNAPSHOTS_DIR [--log]`

- `INPUT_DIR` is the path to the folder containing input images
- `GT_DIR` is the path to the folder containing ground truth depth maps
- `OUT_DIR` is the path to the folder to which will be written output depth maps
- `SNAPSHOTS_DIR` is the path to the folder containing .caffemodel files containing trained network models. All models from this folder will be evaluated.
- `--log` switch is used when the depth values that are produced by the network are in log space

## Frameworks/Libraries needed:

* Caffe
* Python2.7: caffe, scipy, scikit-image, numpy, pypng, cv2, Pillow, matplotlib

## Few notes
- input images should be named in a same way as the corresponding ground truths, with difference that input images should have a suffix 'colors', while ground truth images should have a suffix 'depth'. Note that these suffixes should preceed file extension, e.g., 'image1_colors.png' and corresponding depth map 'image1_depth.png'
- along with .caffemodel file, corresponding deploy network definition file has to be placed into SNAPSHOTS_DIR, with the same name as the model file but with different extension 'prototxt' instead of 'caffemodel'
- there will actually be two output folders created, one OUT_DIR and the other OUT_DIR + '_abs'. OUT_DIR contains output depths that are fit using MVN normalization onto ground truth, OUT_DIR + '_abs' contains the raw output depth maps.
- note that you need AlexNet caffemodel for the training of the global context network, gradient network and their joint configuration. It can be downloaded here: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
