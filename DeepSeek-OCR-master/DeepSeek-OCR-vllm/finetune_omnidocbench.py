#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetune DeepSeek-OCR on OmniDocBench dataset.

This script:
1. Loads OmniDocBench JSON annotations
2. Converts annotations to markdown format (ground truth)
3. Creates a training dataset pairing images with markdown targets
4. Finetunes DeepSeek-OCR model using transformers Trainer
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import re
import langid
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from config import CROP_MODE, PROMPT
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


def poly2bbox(poly):
    """Convert polygon coordinates to bounding box [x1, y1, x2, y2]."""
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    return [L, U, R, D]


def text_norm(text):
    """Normalize text by replacing special characters."""
    after_text = replace_repeated_chars(text)
    return after_text.replace('/t', '\t').replace("\\t", '\t').replace('/n', '\n')


def replace_repeated_chars(input_str):
    """Standardize all consecutive characters."""
    input_str = re.sub(r'_{4,}', '____', input_str)
    input_str = re.sub(r' {4,}', '    ', input_str)
    return re.sub(r'([^a-zA-Z0-9])\1{10,}', r'\1\1\1\1', input_str)


class OmniDocBenchDataset(Dataset):
    """Dataset for OmniDocBench finetuning."""
    
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        max_length: int = 8192,
        prompt_template: str = "<image>\n<|grounding|>Convert the document to markdown.",
        table_format: str = 'html'
    ):
        """
        Args:
            json_path: Path to OmniDocBench.json
            images_dir: Directory containing images
            max_length: Maximum sequence length
            prompt_template: Prompt template for the model
            table_format: Table format ('html' or 'latex')
        """
        self.images_dir = images_dir
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.table_format = table_format
        
        # Load and process samples
        print("Loading OmniDocBench dataset...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        # Process samples to create image-markdown pairs
        self.data = self._process_samples()
        print(f"Loaded {len(self.data)} training samples")
    
    def _process_samples(self):
        """Process samples to extract image-markdown pairs."""
        data = []
        
        for sample in tqdm(self.samples, desc="Processing samples"):
            img_name = os.path.basename(sample['page_info']['image_path'])
            img_path = os.path.join(self.images_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Convert annotations to markdown
            markdown = self._annotations_to_markdown(sample, img_path)
            
            if markdown.strip():
                data.append({
                    'image_path': img_path,
                    'markdown': markdown,
                    'page_info': sample['page_info']
                })
        
        return data
    
    def _annotations_to_markdown(self, sample, img_path):
        """Convert OmniDocBench annotations to markdown format."""
        annos = []
        for x in sample['layout_dets']:
            if x.get('order') is not None:
                annos.append(x)
        
        # Handle truncated text blocks
        saved_element_dict = defaultdict(list)
        related_truncated = []
        truncated_all = {}
        
        for relation in sample.get("extra", {}).get("relation", []):
            if relation.get("relation_type") == 'truncated':
                truncated_all[relation["source_anno_id"]] = ""
                truncated_all[relation["target_anno_id"]] = ""
                exist_flag = False
                for merge_list in related_truncated:
                    if (relation["source_anno_id"] in merge_list or 
                        relation["target_anno_id"] in merge_list):
                        merge_list.append(relation["source_anno_id"])
                        merge_list.append(relation["target_anno_id"])
                        exist_flag = True
                if not exist_flag:
                    related_truncated.append([
                        relation["source_anno_id"], 
                        relation["target_anno_id"]
                    ])
        
        merged_annos = []
        for item in annos:
            if item['anno_id'] not in truncated_all.keys():
                merged_annos.append(item)
            else:
                truncated_all[item['anno_id']] = item
        
        # Merge truncated blocks
        for merge_list in related_truncated:
            text_block_list = [truncated_all[key] for key in merge_list]
            sorted_block = sorted(text_block_list, key=lambda x: x['order'])
            text = ""
            for block in sorted_block:
                line_content = block.get('text', '')
                if line_content:
                    if (langid.classify(line_content)[0] == 'en' and 
                        line_content[-1] != "-"):
                        text += f" {line_content}"
                    elif (langid.classify(line_content)[0] == 'en' and 
                          line_content[-1] == "-"):
                        text = text[:-1] + f"{line_content}"
                    else:
                        text += f"{line_content}"
            
            merged_block = {
                "category_type": sorted_block[0]["category_type"],
                "order": sorted_block[0]["order"],
                "anno_id": sorted_block[0]["anno_id"],
                "text": text,
                "merge_list": sorted_block
            }
            merged_annos.append(merged_block)
        
        # Sort by order
        annos = sorted(merged_annos, key=lambda x: x['order'])
        
        # Convert to markdown
        markdown_parts = []
        img = Image.open(img_path).convert('RGB')
        
        for anno in annos:
            sep = '\n\n'
            category = anno.get("category_type", "")
            
            if category == 'figure':
                # For figures, we'll just add a placeholder
                # In practice, you might want to crop and save the figure
                markdown_parts.append("![Figure]")
                markdown_parts.append(sep)
            elif category == 'table':
                table_content = anno.get(self.table_format, '')
                if table_content:
                    markdown_parts.append(table_content)
                    markdown_parts.append(sep)
            elif anno.get('text'):
                if category == 'title':
                    text = text_norm(anno['text'].strip('#').strip())
                    markdown_parts.append('# ' + text)
                    markdown_parts.append(sep)
                else:
                    text = text_norm(anno['text'])
                    markdown_parts.append(text)
                    markdown_parts.append(sep)
            elif anno.get('html'):
                markdown_parts.append(anno['html'])
                markdown_parts.append(sep)
            elif anno.get('latex'):
                markdown_parts.append(anno['latex'])
                markdown_parts.append(sep)
        
        return ''.join(markdown_parts).strip()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        
        # Prepare prompt and target
        prompt = self.prompt_template
        target = item['markdown']
        return {
            'image': image,
            'prompt': prompt,
            'target': target,
            'image_path': item['image_path']
        }


class DataCollatorForOmniDocBench:
    """Data collator for OmniDocBench finetuning."""
    
    def __init__(self, tokenizer, model, processor=None, max_length: int = 8192, crop_mode: bool = True):
        self.tokenizer = tokenizer
        self.model = model
        self.processor = processor
        self.max_length = max_length
        self.crop_mode = crop_mode
        self.ignore_id = getattr(processor, "ignore_id", -100)
        
        # Try to get processor from model if available
        if processor is None and hasattr(model, 'get_processor'):
            try:
                self.processor = model.get_processor()
            except:
                pass
    
    def __call__(self, features):
        """Collate features into batch."""
        images = [f['image'] for f in features]
        prompts = [f['prompt'] for f in features]
        targets = [f['target'] for f in features]
        
        # Process using model's processor if available
        if self.processor is not None:
            batch_inputs = []
            batch_labels = []
            batch_attention = []
            batch_images_seq_mask = []
            batch_images = []
            batch_images_spatial_crop = []

            for img, prompt, target in zip(images, prompts, targets):
                if prompt.strip() != PROMPT.strip():
                    print("Warning: prompt_template differs from config.PROMPT; using config.PROMPT for image tokens.")
                # Build prompt tokens with image placeholders
                img_processed = self.processor.tokenize_with_images(
                    images=[img],
                    bos=True,
                    eos=True,
                    cropping=self.crop_mode
                )[0]

                input_ids = img_processed[0].squeeze(0)
                pixel_values = img_processed[1]
                images_crop = img_processed[2].squeeze(0)
                images_seq_mask = img_processed[3]
                images_spatial_crop = img_processed[4].squeeze(0)

                # Append target tokens (with EOS)
                target_ids = self.tokenizer.encode(target, add_special_tokens=False)
                if self.tokenizer.eos_token_id is not None:
                    target_ids = target_ids + [self.tokenizer.eos_token_id]
                target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)

                full_input_ids = torch.cat([input_ids, target_ids_tensor], dim=0)
                full_images_seq_mask = torch.cat(
                    [images_seq_mask, torch.zeros(len(target_ids_tensor), dtype=torch.bool)],
                    dim=0
                )

                # Truncate to max_length
                if full_input_ids.size(0) > self.max_length:
                    full_input_ids = full_input_ids[: self.max_length]
                    full_images_seq_mask = full_images_seq_mask[: self.max_length]

                labels = torch.full_like(full_input_ids, fill_value=self.ignore_id)
                prompt_len = min(input_ids.size(0), full_input_ids.size(0))
                target_len = min(len(target_ids_tensor), full_input_ids.size(0) - prompt_len)
                if target_len > 0:
                    labels[prompt_len:prompt_len + target_len] = target_ids_tensor[:target_len]

                attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)

                batch_inputs.append(full_input_ids)
                batch_labels.append(labels)
                batch_attention.append(attention_mask)
                batch_images_seq_mask.append(full_images_seq_mask)
                batch_images.append((images_crop, pixel_values))
                batch_images_spatial_crop.append(images_spatial_crop)

            max_len = min(self.max_length, max(x.size(0) for x in batch_inputs))
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            def pad_1d(tensor, pad_value, length):
                if tensor.size(0) >= length:
                    return tensor[:length]
                pad = torch.full((length - tensor.size(0),), pad_value, dtype=tensor.dtype)
                return torch.cat([tensor, pad], dim=0)

            input_ids = torch.stack([pad_1d(x, pad_id, max_len) for x in batch_inputs], dim=0)
            labels = torch.stack([pad_1d(x, self.ignore_id, max_len) for x in batch_labels], dim=0)
            attention_mask = torch.stack([pad_1d(x, 0, max_len) for x in batch_attention], dim=0)
            images_seq_mask = torch.stack([pad_1d(x, False, max_len) for x in batch_images_seq_mask], dim=0)
            images_spatial_crop = torch.stack(batch_images_spatial_crop, dim=0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "images": batch_images,
                "images_seq_mask": images_seq_mask,
                "images_spatial_crop": images_spatial_crop,
            }
        
        raise ValueError("Could not use model processor")


def main():
    parser = argparse.ArgumentParser(description='Finetune DeepSeek-OCR on OmniDocBench')
    parser.add_argument(
        '--model_path',
        type=str,
        default='deepseek-ai/DeepSeek-OCR',
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='Path to OmniDocBench.json'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directory containing OmniDocBench images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints/deepseek-ocr-omnidocbench',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=100,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Evaluate every N steps'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=50,
        help='Log every N steps'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=8192,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.9,
        help='Train/val split ratio'
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(
        args.model_path,
        # _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    
    # Try to get processor from model
    processor = None
    if hasattr(model, 'processor'):
        processor = model.processor
    elif hasattr(model, 'get_processor'):
        try:
            processor = model.get_processor()
        except:
            pass
    
    # If processor not found, try importing from process module
    if processor is None:
        try:
            from process.image_process import DeepseekOCRProcessor
            processor = DeepseekOCRProcessor(tokenizer=tokenizer)
        except ImportError:
            print("Warning: Could not import DeepseekOCRProcessor")
    
    model = model.to(device)

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    if hasattr(model, "get_input_embeddings") and len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))
    
    # Create dataset
    print("Creating dataset...")
    full_dataset = OmniDocBenchDataset(
        json_path=args.json_path,
        images_dir=args.images_dir,
        max_length=args.max_length
    )
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data collator with device
    data_collator = DataCollatorForOmniDocBench(
        tokenizer=tokenizer,
        model=model,
        processor=processor,
        max_length=args.max_length,
        crop_mode=CROP_MODE
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_pin_memory=False,  # Disable pin_memory to avoid device conflicts
        remove_unused_columns=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":

    import torch
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    device_count = torch.cuda.device_count()
    print(f"可用显卡数量: {device_count}")
    for i in range(device_count):
        print(f"--- GPU {i} ---")
        print(f"显卡名称: {torch.cuda.get_device_name(i)}")
        print(f"显卡计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"显卡内存: {torch.cuda.get_device_properties(i).total_memory}")


    torch.cuda.set_device(1)
    print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
    print(f"当前 CUDA 设备内存: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory}")

    main()
