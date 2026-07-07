# Hierarchical YOLO

This repo provides wrappers around the [hierarchical loss library](https://csbrown-noaa.github.io/hierarchical_loss/) to facilitate usage with YOLO-style architectures, especially the [ultralytics implementation](https://docs.ultralytics.com/).
See the included jupyter notebooks for example usage.

# Installation

```
pip install git+https://github.com/csbrown-noaa/hierarchical_yolo.git

```

# Slides

[Presentation slides on this work](https://docs.google.com/presentation/d/1L0I-8miapJ_PC1Vj0XYP9OaVOk4G8uJjCON_hg2ISVI)

# End-to-End Example: MSCOCO

To see the hierarchical architecture in action, you can download the standard MSCOCO dataset, compile a hierarchical workspace, and train a curriculum model.

### 1. Download and Stage the Data

First, download the 2017 MSCOCO validation and training datasets along with their annotations. We will extract them and manually arrange them into a clean "staging" directory that the generic data orchestrator can parse.

*Note: The orchestrator expects the `hierarchy.json` at the root of the staging directory, alongside the COCO JSON splits. It also expects the images to be nested in a directory matching the JSON's basename (e.g., `instances_val2017/images/`) to safely symlink them without redundant downloading.*

```
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

```
python -m hierarchical_yolo.data_orchestrator \
    --source_dir ./coco_staging \
    --workspace_dir ./coco_workspace

```

### 3. Curriculum Training

To train the model, we point the trainer at the compiled `workspace_dir`. The script will automatically locate the taxonomy tree and iteratively train shallow taxonomic depths first before passing the weights down to deeper class branches.

```
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

```
python -m hierarchical_yolo.predict \
    --workspace_dir ./coco_workspace \
    --model_dir ./runs \
    --project_name coco_hierarchical_run \
    --split val \
    --output ./results/hierarchical_preds.json

```


### 5. Apples-to-Apples Evaluation

Standard metrics (like mAP) heavily privilege models trained explicitly for a single evaluation tier. A hierarchical model evaluates the entire taxonomic tree simultaneously, making direct comparison against a "flat" baseline inherently difficult.

To reduce epistemic error and fairly compare models, we provide an "Apples-to-Apples" benchmark suite. This toolbox dynamically masks predictions and collapses ground truths in PyTorch memory, allowing you to mathematically coerce models into matching hypothesis spaces.

*Note: To evaluate flat baseline models natively on their own taxonomy, simply use the standard ultralytics CLI (e.g., `yolo val model=runs/flat_root/weights/best.pt data=coco_workspace/tier_flat_specialists/000/train.yaml split=test`). The scripts below are used to coerce models that do not natively match the target evaluation space.*

#### Test A: The Objectness Benchmark (Root-Level Detection)

This evaluates how well each model architecture acts as a pure binary object detector. It collapses all ground truths and predictions down to the root node (e.g., "Biota" or "object").

Evaluate the deep Flat Model (coerced to objectness):

```
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
    --weights ./runs/flat_original/weights/best.pt \
    --model_type flat \
    --data_yaml ./coco_workspace/master_yolo/train.yaml \
    --split test




```

Evaluate the Hierarchical Model (coerced to root marginal objectness):

```
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
    --weights ./runs/coco_hierarchical_run/weights/best.pt \
    --model_type hierarchical \
    --data_yaml ./coco_workspace/master_yolo/train.yaml \
    --hierarchy_json ./coco_workspace/hierarchy.json \
    --split test




```

*Interpretation:* Compare these outputs against a native run of a model explicitly trained only to detect roots. If the hierarchical model matches or exceeds the flat root model, it indicates that learning complex hierarchical priors acts as a robust regularizer for base object detection, rather than a distraction.

#### Test B: The Specificity Benchmark (Arbitrary Depth Comparison)

This evaluates fine-grained discrimination on an identical hypothesis space. The hierarchical model's predictions are dynamically masked to exactly match the target vocabulary defined by an arbitrary flat baseline model (e.g., Depth 2).

Evaluate the Hierarchical Model masked to Depth 2 constraints:

```
python -m hierarchical_yolo.apples2apples_benchmarks specificity \
    --weights ./runs/coco_hierarchical_run/weights/best.pt \
    --hierarchical_eval_yaml ./coco_workspace/tier_yolo_full_head/002/train.yaml \
    --flat_data_yaml ./coco_workspace/tier_flat_specialists/002/train.yaml \
    --hierarchy_json ./coco_workspace/hierarchy.json \
    --split test




```

*Interpretation:* Compare this output directly against a native standard `yolo val` run of the Flat model explicitly trained at Depth 2. A higher score for the hierarchical model indicates that topological feature-sharing (passing gradients through parent nodes) improves class recall compared to isolated, mutually exclusive flat buckets at that depth.

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
