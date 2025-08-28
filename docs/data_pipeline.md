## FreiHAND Retargeting Whole Pipeline

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
```bash

python convert_frei.py \
    --processed-dataset-path /home/guangqi/wanglab/dex-retargeting/data/freihand_processed/processed_freihand_dataset.pkl \
    --robot-name xhand \
    --retargeting-type position \
    --hand_type right \
    --output-dir /home/guangqi/wanglab/dex-retargeting/data/freihand_processed \
    --save_images

# TODO how to deal with hand type
python convert_frei_all_hand.py \
    --freihand_dataset_path /home/guangqi/wanglab/dex-retargeting/data/freihand \
    --max-workers 1 \
    --save-images
```

### Data visualization
```bash
python visualize_x_embodiment.py --dataset-path <path2retargetted_dataset>
```