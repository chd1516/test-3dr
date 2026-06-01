if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset> <backbone> <rank> <gpu_id>"
  exit 1
fi

DATASET=$1
BACKBONE=$2
RANK=$3
GPU_ID=$4

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Running $DATASET on GPU: $GPU_ID"


# The output directory stores the model and extracted features; if retraining is unnecessary, you can directly run the test script.


# 1. Start training the model
echo "Training..."
python train.py --dataset "$DATASET" --backbone "$BACKBONE" --r "$RANK"

# 2. Extract text features using InternVL
# export HF_ENDPOINT=https://hf-mirror.com  Configure the HuggingFace mirror source (Optional)
#echo "Running text_feats.py..."
#python text_feats.py --dataset "$DATASET" --backbone "$BACKBONE" --r "$RANK"

# 3. Extract CLIP image features
#echo "Running image_feats.py..."
#python image_feats.py --dataset "$DATASET" --backbone "$BACKBONE" --r "$RANK"

# 4. Testing
#echo "Testing..."
#python test.py --dataset "$DATASET" --backbone "$BACKBONE" --r "$RANK"

echo "All done!"