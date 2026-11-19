"""
Quick evaluation script to get test set metrics for v1 model
"""
import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import jiwer

MODEL_PATH = "./trocr-rxhandbd-checkpoints/final_model"
DATA_DIR = "d:/PROJECT1.O/data/RxHandBD-ML"
MAX_TARGET_LENGTH = 32

def load_samples(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["Images"], row["Text"]) for row in reader]

class RxHandBDDataset(Dataset):
    def __init__(self, samples, image_dir: str, processor: TrOCRProcessor):
        self.samples = samples
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

def build_compute_metrics(processor: TrOCRProcessor):
    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = [[tok if tok != -100 else processor.tokenizer.pad_token_id for tok in seq] for seq in label_ids]
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not pairs:
            return {"cer": 1.0, "wer": 1.0}
        preds, labels = zip(*pairs)
        cer = jiwer.cer(list(labels), list(preds))
        wer = jiwer.wer(list(labels), list(preds))
        return {"cer": cer, "wer": wer}
    return compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluating on device: {device}")

print(f"Loading model from: {MODEL_PATH}")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
model.to(device)

test_csv = os.path.join(DATA_DIR, "Test_Labels.csv")
test_images = os.path.join(DATA_DIR, "Test_Set")
test_samples = load_samples(test_csv)
test_dataset = RxHandBDDataset(test_samples, test_images, processor)

print(f"Test set size: {len(test_samples)}")

training_args = Seq2SeqTrainingArguments(
    output_dir="./temp_eval",
    per_device_eval_batch_size=4,
    fp16=(device == "cuda"),
    predict_with_generate=True,
    report_to=[],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=build_compute_metrics(processor),
)

print("\nRunning test set evaluation...")
test_metrics = trainer.evaluate()
print("\n" + "=" * 60)
print("STAGE B V1 - TEST SET RESULTS:")
print(f"  Character Error Rate (CER): {test_metrics['eval_cer']:.4f}")
print(f"  Word Error Rate (WER):      {test_metrics['eval_wer']:.4f}")
print("=" * 60)
