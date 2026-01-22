# GLFF

### Set up
 ```
 conda create -n glff python=3.10 -y
 pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
 pip install -r requirements.txt
 ```

 ### Train Stage 1 (Segmentation model)
 -Single GPU
 ```
 python S1_train_seg.py
 ```
 -Multi GPU
 ```
 torchrun --nproc_per_node=4 S1_train_seg.py
 ```

### Train Stage 2 (GLFF model)
 ```
 python S2_train_glff.py 
 ```

### Inference whole framework
 ```
 python eval_glff.py
 ```