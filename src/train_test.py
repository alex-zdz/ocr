import re
import unicodedata
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def process_image(image, model, messages_template, processor):
    messages = deepcopy(messages_template)
    messages[0]["content"][0]["image"] = image
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


import matplotlib.pyplot as plt
from IPython.display import display, HTML
import difflib
import pandas as pd
import warnings

# Silence the specific warning about pad_token_id
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")

def highlight_diff(text1, text2):
    """Highlight differences between two strings, returning HTML with blue highlighting"""
    d = difflib.Differ()
    diff = list(d.compare(text1, text2))
    
    text1_chars = []
    text2_chars = []
    
    for i, s in enumerate(diff):
        if s.startswith('  '):  # Common character
            text1_chars.append(f'<span>{s[2:]}</span>')
            text2_chars.append(f'<span>{s[2:]}</span>')
        elif s.startswith('- '):  # In text1 but not text2
            text1_chars.append(f'<span style="background-color:skyblue">{s[2:]}</span>')
        elif s.startswith('+ '):  # In text2 but not text1
            text2_chars.append(f'<span style="background-color:skyblue">{s[2:]}</span>')
    
    return ''.join(text1_chars), ''.join(text2_chars)

def display_results(df, num_examples=5):
    """Display images with ground truth and prediction text"""
    for i in range(min(num_examples, len(df))):
        example = df.iloc[i]
        
        # Get text and highlight differences
        ground_truth = example['ground_truth']
        prediction = example['prediction']
        identifier = example['identifier']
        
        gt_html, pred_html = highlight_diff(ground_truth, prediction)
        
        # Try to retrieve the image if available in the DataFrame
        img = None
        if 'image' in example:
            img = example['image']
        
        # Display image if available
        if img is not None:
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"ID: {identifier}")
            plt.tight_layout()
            plt.show()
        
        # Display the text with highlighted differences using HTML
        display(HTML(f"""
        <div style="font-size: 16px; font-family: Arial, sans-serif;">
            <p><b>ID:</b> {identifier}</p>
            <p><b>Ground Truth:</b> {gt_html}</p>
            <p><b>Prediction:</b> {pred_html}</p>
            <hr>
        </div>
        """))


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i  # Deletion
    for j in range(n + 1):
        dp[0][j] = j  # Insertion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]

def compute_mean_CER(gt_texts, pred_texts, normalize=False):
    total_cer = 0
    num_samples = len(gt_texts)
    for gt_text, pred_text in zip(gt_texts, pred_texts):
        if normalize:
            gt_text = str(gt_text)
            gt_text = unicodedata.normalize('NFKC', gt_text)
            gt_text = re.sub(r'\s+', ' ', gt_text).strip()
            pred_text = str(pred_text)
            pred_text = unicodedata.normalize('NFKC', pred_text)
            pred_text = re.sub(r'\s+', ' ', pred_text).strip()
        gt_chars = list(gt_text)
        pred_chars = list(pred_text)
        distance = levenshtein_distance(gt_chars, pred_chars)
        cer = distance / len(gt_chars) if len(gt_chars) > 0 else 0
        total_cer += cer
    mean_cer = total_cer / num_samples if num_samples > 0 else 0
    return mean_cer


def compute_CER(gt_text, pred_text, normalize=True):
        if normalize:
            gt_text = str(gt_text)
            gt_text = unicodedata.normalize('NFKC', gt_text)
            gt_text = re.sub(r'\s+', ' ', gt_text).strip()
            pred_text = str(pred_text)
            pred_text = unicodedata.normalize('NFKC', pred_text)
            pred_text = re.sub(r'\s+', ' ', pred_text).strip()
        gt_chars = list(gt_text)
        pred_chars = list(pred_text)
        distance = levenshtein_distance(gt_chars, pred_chars)
        cer = distance / len(gt_chars) if len(gt_chars) > 0 else 0
        return cer


def run_inference_and_calculate_cer(model_path, message, dataset):
    # Load the model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="auto").half()
    processor = AutoProcessor.from_pretrained(model_path)

    message_template = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": None,  # Placeholder for image, to be updated
                },
                {"type": "text", "text": f"{message}"},
            ],
        }
    ]

    results = []
    # Run inference on the dataset
    for item in dataset:
        prediction = process_image(item["image"], model, message_template, processor)
        cer = compute_CER(item["text"], prediction, normalize=True)
        results.append({
            "ground_truth": item["text"],
            "num_lines": item["num_lines"],
            "prediction": prediction,
            "identifier": item["identifier"],
            "CER": cer
        })

    # Create DataFrame from results
    df = pd.DataFrame(results)

    return df
