# BiomedParse Quick Start Guide

**Goal**: Replace Qwen with BiomedParse for brain tumor visual grounding using Docker.

---

## 🚀 Quick Start (5 Steps)

### Step 1: Download BiomedParse Docker Image

```bash
# Install gdown for Google Drive downloads
pip install gdown

# Download the Docker image (~5-10 GB, takes 10-30 min depending on connection)
cd /Disk1/afrouz/Projects
gdown https://drive.google.com/uc?id=1eUAY1qvEzM0Ut0PA9BGp6gexn5TiFWj8

# You should now have a file like "biomedparse_docker.tar" or similar
ls -lh *.tar
```

**Alternative**: Download manually from browser:
- Go to: https://drive.google.com/file/d/1eUAY1qvEzM0Ut0PA9BGp6gexn5TiFWj8/view?usp=sharing
- Click "Download"
- Save to `/Disk1/afrouz/Projects/`

### Step 2: Load Docker Image

```bash
cd /Disk1/afrouz/Projects

# Load the image (replace with actual filename)
docker load < biomedparse_docker.tar

# Verify it loaded
docker images | grep biomedparse

# Expected output:
# biomedparse    latest    <image_id>    <size>
```

### Step 3: Inspect Container (Find Inference Script)

```bash
cd /Disk1/afrouz/Projects/TextBraTS/src/spatial_prompting/visual_grounding

# Run inspection to find the inference script location
python biomedparse_docker.py --inspect
```

This will show you:
- Where the inference script is located inside the container
- What arguments it expects
- Directory structure

**Update the script**: Based on inspection output, update the `inference_script` parameter in [biomedparse_docker.py](biomedparse_docker.py:59) (line 59).

### Step 4: Test on Single Case

```bash
# Test inference on a BraTS sample
python biomedparse_docker.py \
    --input /Disk1/afrouz/Data/Merged/BraTS20_Training_002/BraTS20_Training_002_flair.nii \
    --text_prompt "brain tumor" \
    --output outputs/test_biomedparse \
    --gpu 1
```

Check results:
```bash
ls -la outputs/test_biomedparse/
cat outputs/test_biomedparse/results.json  # If exists
```

### Step 5: Process Full Dataset (Optional)

```bash
# Batch process all BraTS cases
python biomedparse_docker.py \
    --batch /Disk1/afrouz/Data/Merged \
    --text_prompt "brain tumor" \
    --output outputs/biomedparse_batch \
    --gpu 1
```

---

## 📋 What You Have Now

After following the Inference Plan, you wanted to replace Qwen with BiomedParse. Here's what's been created:

### Files Created

1. **[biomedparse_vg.py](biomedparse_vg.py)** - Native Python implementation (requires manual setup)
2. **[biomedparse_docker.py](biomedparse_docker.py)** - Docker wrapper (recommended, easier)
3. **[BIOMEDPARSE_DOCKER_SETUP.md](BIOMEDPARSE_DOCKER_SETUP.md)** - Detailed Docker setup guide
4. **[QUICKSTART.md](QUICKSTART.md)** - This quick start guide

### Comparison: Qwen vs BiomedParse

| Feature | Qwen (Old) | BiomedParse (New) |
|---------|-----------|-------------------|
| **Training Data** | General vision-language | Biomedical images (CT, MRI, etc.) |
| **Medical Focus** | ❌ Generic | ✅ Medical-specific |
| **3D Support** | Slice-by-slice only | Native 3D with context |
| **Output** | Text boxes (parsing needed) | Masks + Bboxes + Scores |
| **Performance on Medical** | Poor (as you found) | State-of-the-art |
| **Setup** | Complex (transformers) | Docker (pre-configured) |

---

## 🔧 Troubleshooting

### Issue: Docker image filename unknown

After downloading from Google Drive, the file might have a generic name. Check:

```bash
ls -lh /Disk1/afrouz/Projects/*.tar
# or
ls -lh ~/Downloads/*.tar
```

Then load with the actual filename:
```bash
docker load < <actual_filename>.tar
```

### Issue: Docker not installed

```bash
# Install Docker
sudo apt update
sudo apt install docker.io

# Add yourself to docker group
sudo usermod -aG docker $USER
newgrp docker

# Test
docker --version
```

### Issue: NVIDIA Docker runtime missing

```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Issue: Inference script not found

After `--inspect`, you might find the script is at a different location like:
- `/workspace/inference.py`
- `/code/run_inference.py`
- `/biomedparse/demo.py`

Update line 59 in [biomedparse_docker.py](biomedparse_docker.py:59):
```python
inference_script: str = "/actual/path/to/inference.py",  # Update this
```

### Issue: Permission denied on mounted volumes

```bash
# Fix permissions
sudo chmod -R 755 /Disk1/afrouz/Data/Merged
sudo chmod -R 755 /Disk1/afrouz/Projects/TextBraTS/outputs
```

---

## 🎯 Next Steps After Setup

Once BiomedParse is working:

1. **Compare with Qwen results**:
   ```bash
   # Old Qwen results
   ls visualizations_flair/

   # New BiomedParse results
   ls outputs/test_biomedparse/
   ```

2. **Integrate into your pipeline**:
   - Update your spatial prompting code to use BiomedParse outputs
   - Replace [vg.py](vg.py) references with `biomedparse_docker.py`

3. **Evaluate on full dataset**:
   ```bash
   python biomedparse_docker.py --batch /Disk1/afrouz/Data/Merged --output batch_results/
   ```

4. **Use for visual grounding loss**:
   - Extract bounding boxes from BiomedParse results
   - Feed into your spatial loss computation
   - Compare segmentation performance

---

## 📊 Expected Output Format

BiomedParse should produce better results than Qwen because it:

1. **Understands medical anatomy** - Trained on MRI, CT, etc.
2. **Outputs structured data** - No string parsing needed
3. **Provides confidence scores** - Filter low-quality detections
4. **Native 3D processing** - Better context than slice-by-slice

Example output:
```json
{
  "bbox_3d": [120, 95, 70, 180, 150, 110],
  "confidence": 0.89,
  "mask_shape": [240, 240, 155],
  "detected_structures": ["tumor", "edema"]
}
```

---

## 💡 Tips

1. **Start with FLAIR modality** - Best tumor contrast for MRI
2. **Try different text prompts**:
   - "brain tumor"
   - "glioblastoma"
   - "enhancing lesion"
   - "tumor with edema"

3. **Use GPU 1** to avoid interfering with other work on GPU 0

4. **Check Docker logs** if inference fails:
   ```bash
   docker logs <container_id>
   ```

---

## 📚 Resources

- **Docker Image**: https://drive.google.com/file/d/1eUAY1qvEzM0Ut0PA9BGp6gexn5TiFWj8/view?usp=sharing
- **BiomedParse GitHub**: https://github.com/microsoft/BiomedParse
- **Documentation**: https://microsoft.github.io/BiomedParse/
- **Your Inference Plan**: [/Disk1/afrouz/Projects/TextBraTS/src/INFERENCE_PLAN.md](../../INFERENCE_PLAN.md)

---

**Ready to go!** Start with Step 1 above. 🚀
