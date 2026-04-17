# Hierarchical YOLO

This repo provides wrappers around the [hierarchical loss library](https://csbrown-noaa.github.io/hierarchical_loss/) to facilitate usage with YOLO-style architectures, especially the [ultralytics implementation](https://docs.ultralytics.com/).
See the included jupyter notebooks for example usage.

# Installation

```bash
pip install git+https://github.com/csbrown-noaa/hierarchical_yolo.git
```

# End-to-End Example: MSCOCO

To see the hierarchical architecture in action, you can download the standard MSCOCO dataset, compile a hierarchical workspace, and train a curriculum model.

### 1. Download and Stage the Data

First, download the 2017 MSCOCO validation and training datasets along with their annotations. We will extract them and manually arrange them into a clean "staging" directory that the generic data orchestrator can parse.

*Note: The orchestrator expects the `hierarchy.json` at the root of the staging directory, alongside the COCO JSON splits. It also expects the images to be nested in a directory matching the JSON's basename (e.g., `instances_val2017/images/`) to safely symlink them without redundant downloading.*

```bash
# Create a clean staging directory
mkdir coco_staging
cd coco_staging

# Download COCO Images (Warning: Train is ~18GB, Val is ~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/train2017.zip

# Download COCO Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Download Custom Taxonomy Tree
wget https://raw.githubusercontent.com/csbrown-noaa/hierarchical_yolo/refs/heads/main/hierarchical_yolo/models/coco_hierarchy.json -O hierarchy.json

# Unzip data
unzip val2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip

# Move JSON splits to the root of the staging directory
mv annotations/instances_val2017.json ./val.json
mv annotations/instances_train2017.json ./train.json

# Move the unzipped images into the expected pycocowriter structure
mkdir val
mv val2017 val/images

mkdir train
mv train2017 train/images

# Go back to the repository root
cd ..
```

### 2. Compile the Workspace

Use the `data_orchestrator` module to parse the staging directory. This generic compiler acts as a safety net: it automatically checks the taxonomy, re-indexes any sparse category IDs into a dense, contiguous tensor space, converts the annotations to YOLO format, and builds the hierarchical curriculum datasets.

```bash
python -m hierarchical_yolo.data_orchestrator \
    --source_dir ./coco_staging \
    --workspace_dir ./coco_workspace
```

### 3. Curriculum Training

To train the model, we point the trainer at the compiled `workspace_dir`. The script will automatically locate the taxonomy tree and iteratively train shallow taxonomic depths first before passing the weights down to deeper class branches.

```bash
# Run curriculum training
python -m hierarchical_yolo.train \
    --workspace_dir ./coco_workspace \
    --model_dir ./runs \
    --project_name coco_hierarchical_run \
    --shallow_epochs 2 \
    --final_epochs 20 \
    --imgsz 640
```

### 4. Hierarchical Prediction

Once the model is trained, evaluate the model natively on the test/val set. The predictor uses the same `workspace_dir` to load the taxonomy tree and automatically targets the un-rolled, raw leaf nodes in the master dataset for evaluation.

This uses a biology-aware greedy path traversal to predict the deepest confident node in the taxonomy without letting visually ambiguous objects force a "wild guess".

```bash
python -m hierarchical_yolo.predict \
    --workspace_dir ./coco_workspace \
    --model_dir ./runs \
    --project_name coco_hierarchical_run \
    --split val \
    --output ./results/hierarchical_preds.json
```

# Contributing

We would love to have your contributions that improve current functionality, fix bugs, or add new features.  See [the contributing guidelines](CONTRIBUTING.md) for more info.

# Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and
Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project
code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the
Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub
project will be governed by all applicable Federal law. Any reference to specific commercial products,
processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or
imply their endorsement, recommendation or favoring by the Department of Commerce. The Department
of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to
imply endorsement of any commercial product or activity by DOC or the United States Government.

