# Finetuning DeepSeek-OCR on OmniDocBench

This guide explains how to finetune DeepSeek-OCR on the OmniDocBench dataset.

## Overview

The `finetune_omnidocbench.py` script provides a framework for finetuning DeepSeek-OCR on the OmniDocBench benchmark. It:

1. Loads OmniDocBench JSON annotations
2. Converts annotations to markdown format (ground truth)
3. Creates a training dataset pairing images with markdown targets
4. Finetunes the model using HuggingFace Transformers Trainer

## Prerequisites

1. **Install dependencies:**
```bash
pip install torch transformers pillow tqdm langid
```

2. **Download OmniDocBench dataset:**
   - Download from [Hugging Face](https://huggingface.co/datasets/opendatalab/OmniDocBench) or [OpenDataLab](https://opendatalab.com/OpenDataLab/OmniDocBench)
   - Extract to a directory with the following structure:
   ```
   OmniDocBench/
   ├── images/          # Image files (.jpg)
   ├── pdfs/            # PDF files (optional)
   └── OmniDocBench.json  # Annotation file
   ```

3. **GPU Requirements:**
   - CUDA-capable GPU with sufficient memory (recommended: 24GB+ VRAM)
   - Flash Attention 2 support (for efficient training)

## Usage

### Basic Usage

```bash
python finetune_omnidocbench.py \
    --json_path /path/to/OmniDocBench.json \
    --images_dir /path/to/OmniDocBench/images \
    --output_dir ./checkpoints/deepseek-ocr-omnidocbench \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-5
```

### Full Example with All Options

```bash
python finetune_omnidocbench.py \
    --model_path deepseek-ai/DeepSeek-OCR \
    --json_path /path/to/OmniDocBench.json \
    --images_dir /path/to/OmniDocBench/images \
    --output_dir ./checkpoints/deepseek-ocr-omnidocbench \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 50 \
    --max_length 8192 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --train_split 0.9
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | `deepseek-ai/DeepSeek-OCR` | Path to pretrained model |
| `--json_path` | str | **required** | Path to OmniDocBench.json |
| `--images_dir` | str | **required** | Directory containing images |
| `--output_dir` | str | `./checkpoints/...` | Output directory for checkpoints |
| `--num_epochs` | int | 3 | Number of training epochs |
| `--batch_size` | int | 1 | Training batch size |
| `--learning_rate` | float | 2e-5 | Learning rate |
| `--warmup_steps` | int | 100 | Number of warmup steps |
| `--save_steps` | int | 500 | Save checkpoint every N steps |
| `--eval_steps` | int | 500 | Evaluate every N steps |
| `--logging_steps` | int | 50 | Log every N steps |
| `--max_length` | int | 8192 | Maximum sequence length |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |
| `--fp16` | flag | False | Use mixed precision (FP16) |
| `--bf16` | flag | False | Use bfloat16 precision |
| `--train_split` | float | 0.9 | Train/validation split ratio |

## Important Notes

### 1. Model Architecture Considerations

DeepSeek-OCR uses a custom multimodal architecture with specialized image processing. The current script provides a basic framework, but you may need to:

- **Customize the forward pass**: The model processes images through a specific pipeline (SAM + CLIP + projector). Ensure the training loop properly handles multimodal inputs.

- **Image Processing**: The model uses `tokenize_with_images()` which handles image cropping and tiling. The data collator attempts to use this, but full integration may require additional work.

### 2. Memory Management

- **Batch Size**: Start with `batch_size=1` and increase if GPU memory allows
- **Gradient Accumulation**: Use `--gradient_accumulation_steps` to simulate larger batch sizes
- **Mixed Precision**: Use `--fp16` or `--bf16` to reduce memory usage

### 3. Training Strategy

- **Learning Rate**: Start with `2e-5` and adjust based on loss curves
- **Warmup**: Use warmup steps to stabilize training
- **Checkpointing**: Save checkpoints frequently to avoid losing progress

### 4. Dataset Processing

The script converts OmniDocBench annotations to markdown format:
- Handles truncated text blocks (merges them)
- Converts tables to HTML format (default) or LaTeX
- Preserves reading order
- Normalizes text

### 5. Limitations and Future Work

The current implementation is a starting point. For production use, consider:

1. **Full Multimodal Integration**: Properly integrate the model's image processor into the training loop
2. **Custom Loss Function**: Implement loss masking for prompt tokens (only compute loss on target markdown)
3. **Data Augmentation**: Add image augmentation (rotation, scaling, etc.)
4. **Evaluation Metrics**: Integrate OmniDocBench evaluation metrics during training
5. **Distributed Training**: Add support for multi-GPU training

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use `fp16` or `bf16`
- Reduce `max_length`

### Import Errors

If you encounter import errors for `DeepseekOCRProcessor`:
- Ensure you're running from the correct directory
- Check that all dependencies are installed
- The script will fall back to basic tokenization if the processor can't be imported

### Slow Training

- Use mixed precision (`--fp16` or `--bf16`)
- Increase `gradient_accumulation_steps` to use larger effective batch sizes
- Ensure Flash Attention 2 is properly installed

## Evaluation

After training, evaluate your model using the OmniDocBench evaluation script:

```bash
cd benchmarks/OmniDocBench
python pdf_validation.py --config configs/end2end.yaml
```

Update the config file to point to your model's predictions.

## Citation

If you use OmniDocBench in your research, please cite:

```bibtex
@misc{ouyang2024omnidocbenchbenchmarkingdiversepdf,
    title={OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations}, 
    author={Linke Ouyang and Yuan Qu and Hongbin Zhou and Jiawei Zhu and Rui Zhang and Qunshu Lin and Bin Wang and Zhiyuan Zhao and Man Jiang and Xiaomeng Zhao and Jin Shi and Fan Wu and Pei Chu and Minghao Liu and Zhenxiang Li and Chao Xu and Bo Zhang and Botian Shi and Zhongying Tu and Conghui He},
    year={2024},
    eprint={2412.07626},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.07626}
}
```

## License

Please refer to the original DeepSeek-OCR and OmniDocBench licenses.
