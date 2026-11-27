# Qwen3.5-0.8B on LIBERO

This repository includes LoRA checkpoints for four LIBERO suites. Start with the [one-time setup](#one-time-setup), then choose a workflow:

- [Evaluate a bundled checkpoint](#evaluate-a-bundled-checkpoint) to run the policy in LIBERO. The checkpoints are included, so you can go straight to evaluation without downloading demonstrations.
- [Train and evaluate your own checkpoint](#train-and-evaluate-your-own-checkpoint) to fine-tune Qwen3.5-0.8B on a suite’s demonstrations and evaluate the resulting adapter.

## One-time setup

Run the commands from the `libero-finetune` directory in one Bash session on a Linux machine with an NVIDIA GPU. The driver must support the CUDA builds below. Headless simulation requires EGL. Evaluation uses both Conda environments: `easg-libero` runs the simulator and `easg-qwen` predicts actions.

Choose `libero_spatial`, `libero_goal`, `libero_object`, or `libero_10` for `SUITE`. The commands below clone the simulator source into `third_party/LIBERO` and download the base model into `models/`.

```bash
export SUITE=libero_spatial
conda create -n easg-libero python=3.11 pip -y
conda activate easg-libero
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-sim.txt
export LIBERO_ROOT="$PWD/third_party/LIBERO"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
git -C "$LIBERO_ROOT" checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
python -m pip install --no-deps -e "$LIBERO_ROOT"

conda create -n easg-qwen python=3.11 pip -y
conda activate easg-qwen
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-model.txt
export MODEL_PYTHON="$(command -v python)"
export MODEL_PATH="$PWD/models/Qwen3.5-0.8B"
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir "$MODEL_PATH"
```

In a new shell, restore `SUITE`, `LIBERO_ROOT`, and `MODEL_PATH` from the project directory. Activate `easg-qwen` and run `export MODEL_PYTHON="$(command -v python)"` before switching to `easg-libero`.

## Evaluate a bundled checkpoint

```bash
conda activate easg-libero
python scripts/evaluate.py --benchmark "$SUITE"
```

The evaluator loads the checkpoint for the selected suite automatically. For example, `libero_10` uses `checkpoints/long/checkpoint`. By default it runs 16 episodes per task on one GPU, with up to 600 steps per episode. Add `--save-video` to record rollouts.

Results are written to a new run directory under `data/rollouts/evaluate/`. Open its `summary.json` to check `num_tasks_completed`, `worker_failures`, and `overall_success_rate`.

## Train and evaluate your own checkpoint

### 1. Prepare the demonstrations

Download the HDF5 demonstrations for your suite. LIBERO distributes `libero_10` in the `libero_100` archive. If prompted to customize the dataset directory, enter `n` to use the default.

```bash
conda activate easg-libero
DOWNLOAD_SUITE="$SUITE"
if [[ "$SUITE" == libero_10 ]]; then DOWNLOAD_SUITE=libero_100; fi
python "$LIBERO_ROOT/benchmark_scripts/download_libero_datasets.py" \
  --datasets "$DOWNLOAD_SUITE" --download-dir "$LIBERO_ROOT/libero/datasets"
```

If you already have demonstrations, start here and set `DEMO_DIR` to the directory containing the suite’s `.hdf5` files. Convert the data and validate the sample manifest and images:

```bash
conda activate easg-libero
DEMO_DIR="$LIBERO_ROOT/libero/datasets/$SUITE"
export DATA_ROOT="$PWD/data/training_data/$SUITE"

python scripts/convert_libero_hdf5.py --input-dir "$DEMO_DIR" --output-dir "$DATA_ROOT"
python scripts/validate_dataset.py "$DATA_ROOT"
```

### 2. Train the adapter

Each suite’s recipe trains a new LoRA adapter on the pretrained base model with a learning rate of `5e-5`, 4 samples per GPU, and 2 gradient accumulation steps. The schedules are 70 epochs for Spatial/Goal, 80 for Object, and 75 for Long (`libero_10`).

```bash
conda activate easg-qwen
export OUTPUT_DIR="$PWD/outputs/$SUITE"
RECIPE="$SUITE" bash scripts/train.sh
```

The launcher uses the GPUs listed in `CUDA_VISIBLE_DEVICES` and defaults to GPU 0 when it is unset. To use a specific allocation, set the variable before launching training or keep the value supplied by your scheduler. Effective batch size is the number of GPUs × `PER_DEVICE_TRAIN_BATCH_SIZE` × `GRADIENT_ACCUMULATION_STEPS`.

### 3. Evaluate the trained adapter

Pass the training output directory to the evaluator so it loads your weights. To evaluate an intermediate checkpoint, use a path such as `$OUTPUT_DIR/checkpoint-500`.

```bash
conda activate easg-libero
python scripts/evaluate.py --benchmark "$SUITE" --adapter-path "$OUTPUT_DIR"
```

The run writes its own `summary.json` under `data/rollouts/evaluate/`, using the same evaluation settings as the bundled checkpoint.
