# GLFF

### Set up
 '''
 pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
 pip install -r requirements.txt

 '''
 ### Train Stage 1 (Segmentation model)
 '''
 cd S1_train/
 torchrun --nproc_per_node=4 train_seg.py
 '''