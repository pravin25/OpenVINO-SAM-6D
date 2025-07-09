### Create conda environment
conda env create -f ov_environment.yaml
conda activate ov_sam6d
### IF conda env create failed, can use requirements.txt to setup the venv
# pip install -r requirements.txt

### Install pointnet2
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
cd ../../../

### Download ISM pretrained model
cd Instance_Segmentation_Model
python download_sam.py
python download_fastsam.py
python download_dinov2.py
cd ../

### Download PEM pretrained model
cd Pose_Estimation_Model
python download_sam6d-pem.py
