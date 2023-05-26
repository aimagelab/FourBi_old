# FourBi

We provide the checkpoint for the [model](https://drive.google.com/file/d/1T8d591FpB63fHsrOCjOUpvuuHxhuep_c/view?usp=share_link) trained on DIBCO18 
 
To run the model on a folder with images, run the following command:
```
python binarize.py <path to checkpoint> --src <path to the test images folder> --dst <path to the output folder>
```
The model is trained on patches, then evaluated and tested on complete documents. We provide the code to create the patches and train the model.
For example, to train on DIBCO18, first download the dataset from http://vc.ee.duth.gr/h-dibco2018/benchmark/. Create a folder, then place the images in a sub-folder named "imgs" and the ground truth in a sub-folder named "gt". Then run the following command:
```
python create_patches.py --path_original <path to the dataset folder> --path_destination <path to the folder where the patches will be saved> --patch_size <size of the patches> --overlap_size <size of the overlap>
```
Then, run the following command:
```
python train.py --datasets <all datasets paths> --validation_dataset <name of the validation dataset> --test_dataset <name of the validation dataset>
```