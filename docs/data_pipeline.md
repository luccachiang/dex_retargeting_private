## FreiHAND Retargeting Whole Pipeline
All the code is under [`example/freihand_retargeting`](../example/freihand_retargeting/).

### Download FreiHAND Dataset
Download from [official website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html). Unzip and put it under `dex-retargeting/data`.

### Installaion
```bash
conda remove -n dexenv_data --all
conda create -n dexenv_data python=3.10 -y && conda activate dexenv_data
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit -y
conda install -c conda-forge awscli=2.17.32 -y
pip install numpy opencv-python sapien matplotlib tqdm tyro transforms3d ipdb cython robot_descriptions yourdfpy viser sapien

pip install -e <path2dex-retargeting>
pip install -e <path2sim_dex_vla> # for xhand assets
```

### Data Augmentation
Only need to run once.
```bash
# data augmentation with interpolation and noise
# TODO mp seems quite slow
python augment_freihand_dataset.py --freihand-dataset-path /home/guangqi/wanglab/dex-retargeting/data/freihand --seed 42 --augmentation_factor 5 --output_dir /home/guangqi/wanglab/dex-retargeting/data/freihand_processed --num_workers 8 --no-use-multiprocessing
```

### Generate X-embodiment Retargeting Data
TODO how to deal with left hand?
```bash
python convert_frei_runner.py \
    --processed-dataset-path /home/guangqi/wanglab/dex-retargeting/data/freihand_processed/processed_freihand_dataset.pkl \
    --robot-name xhand \
    --retargeting-type dexpilot \
    --hand_type right \
    --output-dir /home/guangqi/wanglab/dex-retargeting/data/freihand_retargeting_finger_scaled \
    --save_images

# python convert_frei_jinzhou.py \
#     --freihand-dataset-path /home/guangqi/wanglab/dex-retargeting/data/freihand \
#     --robot-name xhand \
#     --retargeting-type position \
#     --hand_type right \
#     --output-dir /home/guangqi/wanglab/dex-retargeting/data/freihand_retargeting \
#     --save_images

python convert_frei_multihand_runner.py \
    --processed-dataset-path /home/guangqi/wanglab/dex-retargeting/data/freihand_processed/processed_freihand_dataset.pkl \
    --hand-type right \
    --max-workers 16 \
    --output-dir /home/guangqi/wanglab/dex-retargeting/data/freihand_retargeting_finger_scaled \
    --retargeting-type dexpilot
    # --robot-names xhand ability inspire \
    # --save-images
    # --max-samples 10
```

### Data visualization
```bash
python visualize_x_embodiment.py --dataset-path /home/guangqi/wanglab/dex-retargeting/data/freihand_retargeting --hands xhand ability inspire
```

### Get left hand data
TODO This doesn't work so far. Instead, directly apply the right hand retargeting qpos to left hand qpos.
Convert right hand data to left hand data.
```bash
python convert_frei_runner.py \
    --processed-dataset-path /home/guangqi/wanglab/dex-retargeting/data/freihand_processed/processed_freihand_dataset.pkl \
    --robot-name xhand \
    --retargeting-type vector \
    --hand_type left \
    --output-dir /home/guangqi/wanglab/dex-retargeting/data/freihand_left_retargeting \
    --save_images
```