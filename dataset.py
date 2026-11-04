from torch.utils.data import Dataset
from transformers import LlavaOnevisionProcessor
import torch
from typing import Dict, Any
import json


class M3cotDataset(Dataset):
    """Dataset for image-based VQA with train/test modes."""
    
    def __init__(
        self, 
        dataset,
        processor: LlavaOnevisionProcessor,
        max_length: int = 2048,
        is_test: bool = False,
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self):
        return len(self.dataset)

    def _format_question(self, item) -> str:
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        formatted_choices = '\n'.join(
            f"{choice_labels[i]}. {choice}" 
            for i, choice in enumerate(item['choices'])
        )
        return f"Question: {item['question']}\nChoices:\n{formatted_choices}"

    def _format_answer(self, item) -> str:
        return f"<reasoning> {item['rationale']} </reasoning>\nFinal Answer: {item['answer']}"

    def _build_conversation(self, question: str, answer: str) -> list:
        if self.is_test:
            # For test: assistant starts with <reasoning> as prompt
            assistant_content = "<reasoning>"
        else:
            assistant_content = answer

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_content}]
            }
        ]

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.dataset[idx]

        question = self._format_question(item)
        answer = self._format_answer(item)
        image = item['image'].convert('RGB')

        if self.is_test:
            # Only include user turn, then manually append <reasoning>
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True  # adds assistant header without closing it
            )
            prompt += "<reasoning>"  # manually append, no <|im_end|>
        else:
            conversation = self._build_conversation(question, answer)
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False
            )

        processed = self.processor(
            text=prompt,
            images=[image],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {}
        for k, v in processed.items():
            result[k] = v.squeeze(0) if isinstance(v, torch.Tensor) else v

        result['answer'] = item['answer']

        return result
    
    
    
    

SYSTEM_PROMPT_STVQA = (
    "You are a vision-language assistant. "
    "Observe the image, then answer the question using this format:\n"
    "<observe>scene description</observe>\n"
    "<scene>{scene graph JSON}</scene>\n"
    "<think>reasoning</think>\n"
    "<answer>(X) answer</answer>"
)


class STVQADataset(Dataset):
    """
    Dataset for ST-VQA (LightChen2333/M3CoT style).

    Fields used:
        images               — PIL image
        question_with_options — clean question + options (no long instruction prompt)
        answer               — full structured response (scene graph + <answer>X</answer>)
        answer_only          — single letter e.g. "A" (stored for eval)

    is_test=False : full conversation (user + assistant answer) for training
    is_test=True  : user turn only, assistant seeded with "<observe>" for generation
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item  = self.dataset[idx]
        image = item['images'].convert('RGB')

        # <image> token inline so processor sees exactly one image reference
        user_text = f"<image>\n{item['question_with_options']}"

        if self.is_test:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_STVQA}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                }
            ]
            prompt  = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            prompt += "<observe>"   # seed the assistant response

        else:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_STVQA}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item['answer']}]
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=False
            )

        processed = self.processor(
            text=prompt,
            images=[image],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in processed.items()
        }

        # kept for evaluation — never used by collator/trainer
        result['answer']      = item['answer_only']   # "A", "B", "C", "D"
        result['answer_full'] = item['answer']         # full scene graph string

        return result
    
    

from torch.utils.data import Dataset
import torch
from typing import Dict, Any

SYSTEM_PROMPT_MK12 = (
    "You are a vision-language assistant. "
    "Carefully observe the image and solve the problem. "
    "Respond with only the letter of the correct answer: A, B, C, or D.\n"
    "<answer>(X)</answer>"
)


class MMK12Dataset(Dataset):
    """
    Dataset for MMK12 validation (chamber111/VPPO_MMK12_validation).

    Fields used:
        images   — list of PIL images (one or more per sample)
        problem  — question text with <image> token already included
        answer   — single letter "A", "B", "C", or "D"

    is_test=False : full conversation (user + assistant answer) for training
    is_test=True  : user turn only, assistant seeded with "<answer>" for generation
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item   = self.dataset[idx]
        # images is a list — convert all to RGB
        images = [img.convert('RGB') for img in item['images']]

        # problem already contains <image> token — use as-is
        user_text = item['problem']

        # format answer as <answer>(X)</answer>
        answer_text = f"<answer>({item['answer']})</answer>"

        if self.is_test:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_MK12}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
            ]
            prompt  = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            prompt += "<answer>"   # seed the assistant response

        else:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_MK12}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}]
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=False
            )

        processed = self.processor(
            text=prompt,
            images=images,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in processed.items()
        }

        # kept for evaluation
        result['answer'] = item['answer']   # "A", "B", "C", "D"

        return result
    


SYSTEM_PROMPT_SafeTag = (
    "You are a content safety classifier. "
    "Given an image and text, determine whether the combined content is safe or unsafe.\n"
    "Respond with only one word: safe or unsafe.\n"
    "<answer>safe or unsafe</answer>"
)


class SafeTagDataset(Dataset):
    """
    Dataset for SafeTag-VL (XuankunRong/SafeTag-VL-3K).

    Fields used:
        images       — list of PIL images
        prompt       — question text with <image> token already included
        combine_tag  — "safe" or "unsafe" (binary label)

    is_test=False : full conversation (user + assistant answer) for training
    is_test=True  : user turn only, assistant seeded with "<answer>" for generation
    """

    def __init__(self, dataset, processor, max_length=1024, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item   = self.dataset[idx]
        images = [img.convert('RGB') for img in item['images']]

        # prompt already contains <image> — use as-is
        user_text   = item['prompt']
        answer_text = f"<answer>{item['combine_tag']}</answer>"

        if self.is_test:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_SafeTag}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
            ]
            prompt  = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            prompt += "<answer>"   # seed the assistant response

        else:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_SafeTag}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}]
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=False
            )

        processed = self.processor(
            text=prompt,
            images=images,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in processed.items()
        }

        # kept for evaluation
        result['answer'] = item['combine_tag']   # "safe" or "unsafe"

        return result
    
    
    
    


SYSTEM_PROMPT_Sphinx = (
    "You are a visual reasoning assistant. "
    "Observe the image carefully, reason through the pattern, then select the correct answer.\n"
    "Use this format:\n"
    "<reasoning>your reasoning here</reasoning>\n"
    "<answer>(X)</answer>"
)


class VisualSphinxDataset(Dataset):
    """
    Dataset for VisualSphinx-Seeds (VisualSphinx/VisualSphinx-Seeds).

    Fields used:
        image                       — list of PIL images
        english_prompt              — instruction text
        options                     — JSON string e.g. '{"A": "A", "B": "B", "C": "C", "D": "D"}'
        english_explanation_rewrite — chain-of-thought reasoning
        correct_answer              — single letter e.g. "C"

    is_test=False : full conversation (user + assistant reasoning + answer) for training
    is_test=True  : user turn only, assistant seeded with "<reasoning>" for generation

    Note: filter out None explanations before passing dataset:
        dataset = dataset.filter(lambda x: x['english_explanation_rewrite'] is not None)
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def _format_options(self, options_raw) -> str:
        """Parse options JSON string and format as readable choices."""
        try:
            options = json.loads(options_raw) if isinstance(options_raw, str) else options_raw
            return "\n".join(f"{k}. {v}" for k, v in options.items())
        except Exception:
            return str(options_raw)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item   = self.dataset[idx]
        images = [img.convert('RGB') for img in item['image']]

        options_str = self._format_options(item['options'])
        user_text   = f"<image>\n{item['english_prompt']}\n{options_str}"

        # full structured response: reasoning + answer
        answer_text = (
            f"<reasoning>{item['english_explanation_rewrite']}</reasoning>\n"
            f"<answer>({item['correct_answer']})</answer>"
        )

        if self.is_test:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_Sphinx}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
            ]
            prompt  = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            prompt += "<reasoning>"   # seed the assistant response

        else:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_Sphinx}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}]
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=False
            )

        processed = self.processor(
            text=prompt,
            images=images,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in processed.items()
        }

        result['answer'] = item['correct_answer']   # "A", "B", "C", "D"

        return result
    
    
    


SYSTEM_PROMPT_rad = (
    "You are a medical visual question answering assistant. "
    "Carefully observe the medical image and answer the question accurately and concisely.\n"
    "<answer>your answer here</answer>"
)


class VQARadDataset(Dataset):
    """
    Dataset for VQA-RAD (flaviagiammarino/vqa-rad).

    Fields used:
        image    — single PIL image (medical scan)
        question — plain text question (no <image> token)
        answer   — free-form text e.g. "yes", "no", or a description

    is_test=False : full conversation (user + assistant answer) for training
    is_test=True  : user turn only, assistant seeded with "<answer>" for generation
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item  = self.dataset[idx]
        # single image — wrap in list for processor
        image = item['image'].convert('RGB')

        # manually prepend <image> since it's not in the question
        user_text   = f"<image>\n{item['question']}"
        answer_text = f"<answer>{item['answer']}</answer>"

        if self.is_test:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_rad}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
            ]
            prompt  = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            prompt += "<answer>"   # seed the assistant response

        else:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_rad}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}]
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=False
            )

        processed = self.processor(
            text=prompt,
            images=[image],       # processor expects a list
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in processed.items()
        }

        result['answer'] = item['answer'].strip().lower()   # for eval

        return result
    
    
    
    
    
"""
Commonsense 170K Dataset
Training data: https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/ft-training_set/commonsense_170k.json

Format:
    {
        "instruction": "...",
        "input": "",
        "output": "the correct answer is <answer>",
        "answer": "<answer>"
    }
"""

import json
import requests
from torch.utils.data import Dataset

TRAIN_URL = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/ft-training_set/commonsense_170k.json"
)

PROMPT_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def load_json_from_url(url: str) -> list:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def load_json_from_path(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def build_prompt(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction)


class CommonsenseDataset(Dataset):
    """
    Dataset for commonsense_170k training/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        - Prompt tokens  → -100  (masked, no loss)
        - Padding tokens → -100  (masked, no loss)
        - Response tokens → kept (loss computed here only)

    With left padding the sequence looks like:
        [PAD PAD ... PAD | BOS prompt tokens | response tokens]
        <---- masked ---->  <--- masked ---->   <--- trained --->

    We tokenize the response separately (no special tokens) to get
    response_len, then mask everything except the last response_len tokens.

    Args:
        data:        List of dicts with 'instruction', 'input', 'output', 'answer'
        tokenizer:   HuggingFace tokenizer (must have padding_side='left')
        max_length:  Max token length
        is_test:     If True, returns prompt + answer for generation (no labels)
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is X"
        answer      = sample.get("answer", "")

        prompt = build_prompt(instruction, input_text)

        # ── Test mode: return prompt only for generation ──────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence with response-only labels ───────────

        # Tokenise full sequence (prompt + response) with left padding
        full_text = prompt + output
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Tokenise response alone (no special tokens) to get its length
        # This tells us how many real tokens at the END belong to the response
        response_enc = self.tokenizer(
            output,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # With left padding:
        #   [PAD PAD ... | BOS prompt | response]
        #                              ^--- last response_len tokens
        # Mask everything before the last response_len positions
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask any remaining padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def load_commonsense_train(tokenizer, max_length=1000, url=TRAIN_URL, 
                            local_path=None, shuffle=True, seed=42):
    data = load_json_from_path(local_path) if local_path else load_json_from_url(url)
    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(data)
    print(f"  Loaded {len(data):,} training examples")
    return CommonsenseDataset(data, tokenizer, max_length=max_length, is_test=False)


def load_commonsense_test(
    tokenizer,
    url: str,
    max_length: int = 1000,
    local_path: str = None,
) -> CommonsenseDataset:
    """Load a single commonsense test set."""
    if local_path:
        print(f"Loading test data from {local_path}")
        data = load_json_from_path(local_path)
    else:
        print(f"Downloading test data from {url}")
        data = load_json_from_url(url)
    print(f"  Loaded {len(data):,} test examples")
    return CommonsenseDataset(data, tokenizer, max_length=max_length, is_test=True)




"""
ARC-Challenge Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _ARC suffix to avoid conflicts with other datasets.
"""

import json
import requests
from torch.utils.data import Dataset

TRAIN_URL_ARC = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/ARC-Challenge/train.json"
)
TEST_URL_ARC = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/ARC-Challenge/test.json"
)

PROMPT_TEMPLATE_WITH_INPUT_ARC = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_ARC = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_arc(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_arc(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_arc_train(local_path: str = None) -> list:
    """Load ARC-Challenge train split. Returns raw list of dicts."""
    return load_json_from_path_arc(local_path) if local_path else load_json_from_url_arc(TRAIN_URL_ARC)


def load_arc_test(local_path: str = None) -> list:
    """Load ARC-Challenge test split. Returns raw list of dicts."""
    return load_json_from_path_arc(local_path) if local_path else load_json_from_url_arc(TEST_URL_ARC)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_arc(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_ARC.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_ARC.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class ARCDataset(Dataset):
    """
    Dataset for ARC-Challenge train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Args:
        data:       Raw list of dicts loaded via load_arc_train / load_arc_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1000)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is answerX"
        answer      = sample.get("answer", "")

        prompt = build_prompt_arc(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        
"""
ARC-Easy Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _ARCEASY suffix to avoid conflicts with other datasets.
"""


TRAIN_URL_ARCEASY = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/ARC-Easy/train.json"
)
TEST_URL_ARCEASY = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/ARC-Easy/test.json"
)

PROMPT_TEMPLATE_WITH_INPUT_ARCEASY = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_ARCEASY = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_arceasy(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_arceasy(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_arceasy_train(local_path: str = None) -> list:
    """Load ARC-Easy train split. Returns raw list of dicts."""
    return load_json_from_path_arceasy(local_path) if local_path else load_json_from_url_arceasy(TRAIN_URL_ARCEASY)


def load_arceasy_test(local_path: str = None) -> list:
    """Load ARC-Easy test split. Returns raw list of dicts."""
    return load_json_from_path_arceasy(local_path) if local_path else load_json_from_url_arceasy(TEST_URL_ARCEASY)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_arceasy(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_ARCEASY.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_ARCEASY.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class ARCEasyDataset(Dataset):
    """
    Dataset for ARC-Easy train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Args:
        data:       Raw list of dicts loaded via load_arceasy_train / load_arceasy_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1000)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is answerX"
        answer      = sample.get("answer", "")

        prompt = build_prompt_arceasy(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        
"""
BoolQ Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _BOOLQ suffix to avoid conflicts with other datasets.
"""


TRAIN_URL_BOOLQ = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/boolq/train.json"
)
TEST_URL_BOOLQ = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/boolq/test.json"
)

PROMPT_TEMPLATE_WITH_INPUT_BOOLQ = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_BOOLQ = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_boolq(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_boolq(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_boolq_train(local_path: str = None) -> list:
    """Load BoolQ train split. Returns raw list of dicts."""
    return load_json_from_path_boolq(local_path) if local_path else load_json_from_url_boolq(TRAIN_URL_BOOLQ)


def load_boolq_test(local_path: str = None) -> list:
    """Load BoolQ test split. Returns raw list of dicts."""
    return load_json_from_path_boolq(local_path) if local_path else load_json_from_url_boolq(TEST_URL_BOOLQ)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_boolq(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_BOOLQ.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_BOOLQ.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class BoolQDataset(Dataset):
    """
    Dataset for BoolQ train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Args:
        data:       Raw list of dicts loaded via load_boolq_train / load_boolq_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1000)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is true/false"
        answer      = sample.get("answer", "")

        prompt = build_prompt_boolq(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        

"""
HellaSwag Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _HELLASWAG suffix to avoid conflicts with other datasets.
"""



TRAIN_URL_HELLASWAG = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/hellaswag/train.json"
)
TEST_URL_HELLASWAG = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/hellaswag/test.json"
)

PROMPT_TEMPLATE_WITH_INPUT_HELLASWAG = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_HELLASWAG = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_hellaswag(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_hellaswag(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_hellaswag_train(local_path: str = None) -> list:
    """Load HellaSwag train split. Returns raw list of dicts."""
    return load_json_from_path_hellaswag(local_path) if local_path else load_json_from_url_hellaswag(TRAIN_URL_HELLASWAG)


def load_hellaswag_test(local_path: str = None) -> list:
    """Load HellaSwag test split. Returns raw list of dicts."""
    return load_json_from_path_hellaswag(local_path) if local_path else load_json_from_url_hellaswag(TEST_URL_HELLASWAG)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_hellaswag(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_HELLASWAG.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_HELLASWAG.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class HellaSwagDataset(Dataset):
    """
    Dataset for HellaSwag train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Args:
        data:       Raw list of dicts loaded via load_hellaswag_train / load_hellaswag_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1000)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is endingX"
        answer      = sample.get("answer", "")

        prompt = build_prompt_hellaswag(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        
"""
OpenBookQA Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _OPENBOOKQA suffix to avoid conflicts with other datasets.
"""


TRAIN_URL_OPENBOOKQA = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/openbookqa/train.json"
)
TEST_URL_OPENBOOKQA = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/openbookqa/test.json"
)

PROMPT_TEMPLATE_WITH_INPUT_OPENBOOKQA = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_OPENBOOKQA = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_openbookqa(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_openbookqa(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_openbookqa_train(local_path: str = None) -> list:
    """Load OpenBookQA train split. Returns raw list of dicts."""
    return load_json_from_path_openbookqa(local_path) if local_path else load_json_from_url_openbookqa(TRAIN_URL_OPENBOOKQA)


def load_openbookqa_test(local_path: str = None) -> list:
    """Load OpenBookQA test split. Returns raw list of dicts."""
    return load_json_from_path_openbookqa(local_path) if local_path else load_json_from_url_openbookqa(TEST_URL_OPENBOOKQA)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_openbookqa(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_OPENBOOKQA.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_OPENBOOKQA.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class OpenBookQADataset(Dataset):
    """
    Dataset for OpenBookQA train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Args:
        data:       Raw list of dicts loaded via load_openbookqa_train / load_openbookqa_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1000)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is answerX"
        answer      = sample.get("answer", "")

        prompt = build_prompt_openbookqa(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        
        
        
"""
Social_i_QA Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _SOCIALIQA suffix to avoid conflicts with other datasets.
"""


TRAIN_URL_SOCIALIQA = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/social_i_qa/train.json"
)
TEST_URL_SOCIALIQA = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/social_i_qa/test.json"
)

PROMPT_TEMPLATE_WITH_INPUT_SOCIALIQA = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_SOCIALIQA = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_socialiqa(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_socialiqa(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_socialiqa_train(local_path: str = None) -> list:
    """Load Social_i_QA train split. Returns raw list of dicts."""
    return load_json_from_path_socialiqa(local_path) if local_path else load_json_from_url_socialiqa(TRAIN_URL_SOCIALIQA)


def load_socialiqa_test(local_path: str = None) -> list:
    """Load Social_i_QA test split. Returns raw list of dicts."""
    return load_json_from_path_socialiqa(local_path) if local_path else load_json_from_url_socialiqa(TEST_URL_SOCIALIQA)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_socialiqa(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_SOCIALIQA.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_SOCIALIQA.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class SocialIQADataset(Dataset):
    """
    Dataset for Social_i_QA train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Args:
        data:       Raw list of dicts loaded via load_socialiqa_train / load_socialiqa_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1000)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is answerX"
        answer      = sample.get("answer", "")

        prompt = build_prompt_socialiqa(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        
"""
WinoGrande Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _WINOGRANDE suffix to avoid conflicts with other datasets.
"""



TRAIN_URL_WINOGRANDE = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/winogrande/train.json"
)
TEST_URL_WINOGRANDE = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset/winogrande/test.json"
)

PROMPT_TEMPLATE_WITH_INPUT_WINOGRANDE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_WINOGRANDE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_winogrande(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_winogrande(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_winogrande_train(local_path: str = None) -> list:
    """Load WinoGrande train split. Returns raw list of dicts."""
    return load_json_from_path_winogrande(local_path) if local_path else load_json_from_url_winogrande(TRAIN_URL_WINOGRANDE)


def load_winogrande_test(local_path: str = None) -> list:
    """Load WinoGrande test split. Returns raw list of dicts."""
    return load_json_from_path_winogrande(local_path) if local_path else load_json_from_url_winogrande(TEST_URL_WINOGRANDE)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_winogrande(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_WINOGRANDE.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_WINOGRANDE.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class WinoGrandeDataset(Dataset):
    """
    Dataset for WinoGrande train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Args:
        data:       Raw list of dicts loaded via load_winogrande_train / load_winogrande_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1000)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1000,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample["output"]      # "the correct answer is option1/option2"
        answer      = sample.get("answer", "")

        prompt = build_prompt_winogrande(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        
        
"""
Math Reasoning Dataset — add this into dataset.py alongside other dataset classes.

Variable naming uses _MATH suffix to avoid conflicts with other datasets.

Train : math_10k.json  (chain-of-thought outputs)
Tests : AddSub, MultiArith, SVAMP, SingleEq, GSM8K, MathQA, MAWPS
"""


TRAIN_URL_MATH = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/ft-training_set/math_10k.json"
)

TEST_URLS_MATH = {
    "AddSub":    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/AddSub/test.json",
    "MultiArith":"https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/MultiArith/test.json",
    "SVAMP":     "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/SVAMP/test.json",
    "SingleEq":  "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/SingleEq/test.json",
    "GSM8K":     "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/gsm8k/test.json",
    "MathQA":    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/mathqa/test.json",
    "MAWPS":     "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/mawps/test.json",
}

PROMPT_TEMPLATE_WITH_INPUT_MATH = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_TEMPLATE_NO_INPUT_MATH = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Data loading (separate from dataset class) ───────────────────────────────

def load_json_from_url_math(url: str) -> list:
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"  Loaded {len(data):,} examples")
    return data


def load_json_from_path_math(path: str) -> list:
    print(f"  Loading {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")
    return data


def load_math_train(local_path: str = None) -> list:
    """Load math_10k train data. Returns raw list of dicts."""
    return load_json_from_path_math(local_path) if local_path else load_json_from_url_math(TRAIN_URL_MATH)


def load_math_test(name: str, local_path: str = None) -> list:
    """
    Load a single math test set by name.
    name: one of AddSub, MultiArith, SVAMP, SingleEq, GSM8K, MathQA, MAWPS
    """
    if local_path:
        return load_json_from_path_math(local_path)
    if name not in TEST_URLS_MATH:
        raise ValueError(f"Unknown test set '{name}'. Choose from: {list(TEST_URLS_MATH.keys())}")
    return load_json_from_url_math(TEST_URLS_MATH[name])


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt_math(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT_MATH.format(
            instruction=instruction,
            input=input_text,
        )
    return PROMPT_TEMPLATE_NO_INPUT_MATH.format(instruction=instruction)


# ─── Dataset class ────────────────────────────────────────────────────────────

class MathDataset(Dataset):
    """
    Dataset for math reasoning train/test data.
    Text-only — no images. Designed for LLaMA / causal LM with LEFT padding.

    Label masking (train only):
        [PAD PAD ... | BOS | prompt tokens | response tokens | EOS]
        <────────── labels = -100 ────────>  <────── trained ─────>

    Note: math_10k outputs are long chain-of-thought — use max_length >= 1024.

    Args:
        data:       Raw list of dicts loaded via load_math_train / load_math_test
        tokenizer:  HuggingFace tokenizer (padding_side will be set to 'left')
        max_length: Max token length (default 1024)
        is_test:    If True, returns prompt-only + answer string for generation
    """

    def __init__(
        self,
        data: list,
        tokenizer,
        max_length: int = 1024,
        is_test: bool = False,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        instruction = sample["instruction"]
        input_text  = sample.get("input", "")
        output      = sample.get("output", "")   # may be empty in MathQA/MAWPS test
        answer      = sample.get("answer", "")

        prompt = build_prompt_math(instruction, input_text)

        # ── Test mode: prompt only, for generation ────────────────────────
        if self.is_test:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":         answer,
                "prompt":         prompt,
            }

        # ── Train mode: full sequence, response-only labels ───────────────
        # Append EOS so model learns to stop after the answer
        output_with_eos = output + self.tokenizer.eos_token

        full_text = prompt + output_with_eos
        full_enc  = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Response length includes EOS so it is not masked out
        response_enc = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        response_len = response_enc["input_ids"].shape[1]

        # Mask everything except the last response_len tokens
        labels = input_ids.clone()
        labels[:-response_len]      = -100   # mask PAD + prompt
        labels[attention_mask == 0] = -100   # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        
        
"""
Qwen3-VL Dataset Classes — ported from LLaVA dataset.py.

Same system prompts, answer formats, and test-time seeding as LLaVA version.
Only the processor call changes:
    LLaVA:     processor(text=prompt, images=[image], ...)
    Qwen3-VL:  apply_chat_template(messages) + process_vision_info(messages)
               → processor(text=[text], images=image_inputs, ...)

Image is passed inside the messages dict ({"type": "image", "image": pil_image})
instead of as a separate argument, so Qwen3-VL handles dynamic resolution properly.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any
from qwen_vl_utils import process_vision_info


# =============================================================================
#  M3COT
# =============================================================================

class Qwen3VLM3cotDataset(Dataset):
    """
    M3CoT dataset — ported from LLaVA M3cotDataset.
    Fields: image (single), question, choices (list), rationale, answer
    Answer format: <reasoning> {rationale} </reasoning>\nFinal Answer: {answer}
    Test seed: <reasoning>
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def _format_question(self, item) -> str:
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        formatted_choices = '\n'.join(
            f"{choice_labels[i]}. {choice}"
            for i, choice in enumerate(item['choices'])
        )
        return f"Question: {item['question']}\nChoices:\n{formatted_choices}"

    def _format_answer(self, item) -> str:
        return f"<reasoning> {item['rationale']} </reasoning>\nFinal Answer: {item['answer']}"

    def __getitem__(self, idx) -> Dict[str, Any]:
        item     = self.dataset[idx]
        image    = item['image'].convert('RGB')
        question = self._format_question(item)
        answer   = self._format_answer(item)

        if self.is_test:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question},
                ]}
            ]
            text  = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text += "<reasoning>"   # seed assistant response
        else:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question},
                ]},
                {"role": "assistant", "content": answer},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

        image_inputs, _ = process_vision_info(messages)
        processed = self.processor(
            text=[text],
            images=image_inputs,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in processed.items()}
        result['answer'] = item['answer']
        return result


# =============================================================================
#  STVQA
# =============================================================================

SYSTEM_PROMPT_STVQA = (
    "You are a vision-language assistant. "
    "Observe the image, then answer the question using this format:\n"
    "<observe>scene description</observe>\n"
    "<scene>{scene graph JSON}</scene>\n"
    "<think>reasoning</think>\n"
    "<answer>(X) answer</answer>"
)


class Qwen3VLSTVQADataset(Dataset):
    """
    STVQA dataset — ported from LLaVA STVQADataset.
    Fields: images (single image stored as item, not list), question_with_options,
            answer (full scene graph), answer_only (single letter A/B/C/D)
    Test seed: <observe>
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item  = self.dataset[idx]
        image = item['images'].convert('RGB')

        question = item['question_with_options']

        if self.is_test:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_STVQA},
                {"role": "user",    "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question},
                ]},
            ]
            text  = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text += "<observe>"   # seed assistant response
        else:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_STVQA},
                {"role": "user",    "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question},
                ]},
                {"role": "assistant", "content": item['answer']},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

        image_inputs, _ = process_vision_info(messages)
        processed = self.processor(
            text=[text],
            images=image_inputs,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in processed.items()}
        result['answer']      = item['answer_only']   # "A", "B", "C", "D"
        result['answer_full'] = item['answer']         # full scene graph string
        return result


# =============================================================================
#  MMK12
# =============================================================================

SYSTEM_PROMPT_MK12 = (
    "You are a vision-language assistant. "
    "Carefully observe the image and solve the problem. "
    "Respond with only the letter of the correct answer: A, B, C, or D.\n"
    "<answer>(X)</answer>"
)


class Qwen3VLMMK12Dataset(Dataset):
    """
    MMK12 dataset — ported from LLaVA MMK12Dataset.
    Fields: images (list of PIL), problem (question with <image> token), answer (A/B/C/D)
    Answer format: <answer>(X)</answer>
    Test seed: <answer>
    Note: problem contains <image> token — must strip it since Qwen3-VL handles
          images via the messages dict not inline tokens.
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item   = self.dataset[idx]
        images = [img.convert('RGB') for img in item['images']]

        # problem may contain <image> token — strip it, image goes in messages dict
        user_text   = item['problem'].replace('<image>', '').strip()
        answer_text = f"<answer>({item['answer']})</answer>"

        # Build content with all images first, then text
        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": user_text})

        if self.is_test:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_MK12},
                {"role": "user",    "content": user_content},
            ]
            text  = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text += "<answer>"   # seed assistant response
        else:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_MK12},
                {"role": "user",    "content": user_content},
                {"role": "assistant", "content": answer_text},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

        image_inputs, _ = process_vision_info(messages)
        processed = self.processor(
            text=[text],
            images=image_inputs,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in processed.items()}
        result['answer'] = item['answer']   # "A", "B", "C", "D"
        return result


# =============================================================================
#  SAFETAG
# =============================================================================

SYSTEM_PROMPT_SafeTag = (
    "You are a content safety classifier. "
    "Given an image and text, determine whether the combined content is safe or unsafe.\n"
    "Respond with only one word: safe or unsafe.\n"
    "<answer>safe or unsafe</answer>"
)


class Qwen3VLSafeTagDataset(Dataset):
    """
    SafeTag dataset — ported from LLaVA SafeTagDataset.
    Fields: images (list of PIL), prompt (with <image> token), combine_tag (safe/unsafe)
    Answer format: <answer>{combine_tag}</answer>
    Test seed: <answer>
    Note: prompt contains <image> token — strip it.
    """

    def __init__(self, dataset, processor, max_length=1024, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item   = self.dataset[idx]
        images = [img.convert('RGB') for img in item['images']]

        # prompt contains <image> — strip it
        user_text   = item['prompt'].replace('<image>', '').strip()
        answer_text = f"<answer>{item['combine_tag']}</answer>"

        # Build content with all images first, then text
        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": user_text})

        if self.is_test:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_SafeTag},
                {"role": "user",    "content": user_content},
            ]
            text  = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text += "<answer>"   # seed assistant response
        else:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_SafeTag},
                {"role": "user",    "content": user_content},
                {"role": "assistant", "content": answer_text},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

        image_inputs, _ = process_vision_info(messages)
        processed = self.processor(
            text=[text],
            images=image_inputs,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in processed.items()}
        result['answer'] = item['combine_tag']   # "safe" or "unsafe"
        return result


# =============================================================================
#  VQARAD
# =============================================================================

SYSTEM_PROMPT_rad = (
    "You are a medical visual question answering assistant. "
    "Carefully observe the medical image and answer the question accurately and concisely.\n"
    "<answer>your answer here</answer>"
)


class Qwen3VLVQARadDataset(Dataset):
    """
    VQA-RAD dataset — ported from LLaVA VQARadDataset.
    Fields: image (single PIL), question, answer
    Answer format: <answer>{answer}</answer>
    Test seed: <answer>
    """

    def __init__(self, dataset, processor, max_length=2048, is_test=False):
        self.dataset    = dataset
        self.processor  = processor
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item  = self.dataset[idx]
        image = item['image'].convert('RGB')

        question    = item['question']
        answer_text = f"<answer>{item['answer']}</answer>"

        if self.is_test:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_rad},
                {"role": "user",    "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question},
                ]},
            ]
            text  = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text += "<answer>"   # seed assistant response
        else:
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT_rad},
                {"role": "user",    "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question},
                ]},
                {"role": "assistant", "content": answer_text},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

        image_inputs, _ = process_vision_info(messages)
        processed = self.processor(
            text=[text],
            images=image_inputs,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in processed.items()}
        result['answer'] = item['answer'].strip().lower()   # for eval
        return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
Generic Qwen3-VL Video QA Dataset for facebook/minimal_video_pairs.

Supports all subsets:
    human_object_interactions
    intuitive_physics
    robot_object_interactions
    temporal_reasoning

All subsets share the same schema:
    video_id, video_path, question, answer, candidates (2-item list), source

Usage:
    from dataset import Qwen3VLVideoDataset, SYSTEM_PROMPTS

    train_dataset = Qwen3VLVideoDataset(
        raw_train,
        processor=processor,
        is_test=False,
        system_prompt=SYSTEM_PROMPTS["temporal_reasoning"],
    )
"""

import os
import ast
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List
from qwen_vl_utils import process_vision_info


# ── Confirmed correct base path ───────────────────────────────────────────────
VIDEO_BASE = (
    "/ocean/projects/cis250089p/nprottasha/tokens"
    "/minimal_video_pairs/videos"
)

# Official idx_map from util.py — only A and B
IDX_MAP = ["a", "b"]

# ── Per-subset system prompts — add new subsets here ─────────────────────────
SYSTEM_PROMPTS = {
    "human_object_interactions": (
        "You are a video question answering assistant. "
        "Watch the video carefully and answer the multiple choice question. "
        "Choose the correct answer and respond with only A or B.\n"
        "<answer>A or B</answer>"
    ),
    "intuitive_physics": (
        "You are a physical plausibility reasoning assistant. "
        "Watch the video carefully and determine whether the events shown are "
        "physically plausible according to intuitive physics principles such as "
        "object permanence, shape constancy, and continuous object trajectories. "
        "Choose the correct answer and respond with only A or B.\n"
        "<answer>A or B</answer>"
    ),
    "robot_object_interactions": (
        "You are a robot manipulation video understanding assistant. "
        "Watch the video carefully and answer the multiple choice question "
        "about robot and object interactions. "
        "Choose the correct answer and respond with only A or B.\n"
        "<answer>A or B</answer>"
    ),
    "temporal_reasoning": (
        "You are a temporal video reasoning assistant. "
        "Watch the video carefully and answer the multiple choice question "
        "about the order and timing of events shown in the video. "
        "Choose the correct answer and respond with only A or B.\n"
        "<answer>A or B</answer>"
    ),
    # Add new subsets here:
    # "new_subset": "Your custom system prompt ...",
}

# Fallback if subset not registered in SYSTEM_PROMPTS
_DEFAULT_SYSTEM_PROMPT = (
    "You are a video question answering assistant. "
    "Watch the video carefully and answer the multiple choice question. "
    "Choose the correct answer and respond with only A or B.\n"
    "<answer>A or B</answer>"
)


# =============================================================================
#  DATASET
# =============================================================================

class Qwen3VLVideoDataset(Dataset):
    """
    Generic Video QA dataset for any facebook/minimal_video_pairs subset.

    All subsets share the same schema:
        video_id, video_path, question, answer, candidates (2-item list), source

    Args:
        dataset       : HuggingFace dataset split (e.g. raw["mini"])
        processor     : Qwen3-VL AutoProcessor
        max_length    : token truncation limit (default 2048)
        is_test       : False = full conversation for training
                        True  = user turn only, seeded with <answer> for generation
        video_base    : root directory containing all subset video folders
        fps           : frames per second to sample from video
        max_frames    : max number of frames to extract
        system_prompt : system prompt string; use SYSTEM_PROMPTS[subset] or pass custom
    """

    def __init__(
        self,
        dataset,
        processor,
        max_length:    int   = 2048,
        is_test:       bool  = False,
        video_base:    str   = VIDEO_BASE,
        fps:           float = 1.0,
        max_frames:    int   = 8,
        system_prompt: str   = _DEFAULT_SYSTEM_PROMPT,
    ):
        self.dataset       = dataset
        self.processor     = processor
        self.max_length    = max_length
        self.is_test       = is_test
        self.video_base    = video_base
        self.fps           = fps
        self.max_frames    = max_frames
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.dataset)

    def _parse_candidates(self, item) -> List[str]:
        raw = item.get('candidates', '[]')
        if isinstance(raw, list):
            return raw
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []

    def _get_answer_idx(self, item) -> str:
        """Return answer index '0' or '1' matching official util.py eval."""
        candidates = self._parse_candidates(item)
        answer     = str(item['answer'])
        try:
            return str(candidates.index(answer))
        except ValueError:
            return "0"   # fallback

    def _get_answer_letter(self, item) -> str:
        """Return 'A' or 'B' for training target."""
        idx = self._get_answer_idx(item)
        return IDX_MAP[int(idx)].upper()   # '0'→'A', '1'→'B'

    def _format_question(self, item) -> str:
        question   = item['question']
        candidates = self._parse_candidates(item)
        if len(candidates) == 2:
            return (f"{question}\n"
                    f"(A) {candidates[0]}\n"
                    f"(B) {candidates[1]}")
        return question

    def __getitem__(self, idx) -> Dict[str, Any]:
        item          = self.dataset[idx]
        video_path    = os.path.join(self.video_base, item['video_path'])
        question      = self._format_question(item)
        answer_letter = self._get_answer_letter(item)
        answer_text   = f"<answer>{answer_letter}</answer>"

        user_content = [
            {
                "type":       "video",
                "video":      video_path,
                "nframes":    self.max_frames,
                "max_pixels": 128 * 32 * 32,
            },
            {"type": "text", "text": question},
        ]

        if self.is_test:
            messages = [
                {"role": "system",    "content": self.system_prompt},
                {"role": "user",      "content": user_content},
            ]
            text  = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text += "<answer>"   # seed assistant turn
        else:
            messages = [
                {"role": "system",    "content": self.system_prompt},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": answer_text},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception as e:
            print(f"  ⚠ Failed to read {video_path}: {e}")
            from PIL import Image
            import numpy as np
            dummy    = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            fallback = [{"role": "user", "content": [
                {"type": "image", "image": dummy},
                {"type": "text",  "text": question},
            ]}]
            image_inputs, video_inputs = process_vision_info(fallback)
            text = self.processor.apply_chat_template(
                fallback, tokenize=False,
                add_generation_prompt=self.is_test)
            if self.is_test:
                text += "<answer>"

        processed = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in processed.items()}

        # Metadata for evaluator — never passed to the model
        result['answer']      = answer_letter                  # "A" or "B"
        result['answer_text'] = item['answer']                 # original answer string
        result['video_id']    = item.get('video_id', '')
        result['candidates']  = str(self._parse_candidates(item))

        return result
    
    
    

    
    # =============================================================================
#  ADD THESE TO YOUR dataset.py
# =============================================================================

# Make sure these are at the top of dataset.py if not already there:
#   import json
#   import os
#   from torch.utils.data import Dataset
#   from qwen_vl_utils import process_vision_info


def load_videomathqa(hf_dataset, test_size=0.2, seed=42):
    """
    VideoMathQA only has a 'test' split on HuggingFace.
    Splits it 80/20 into train and test.

    Usage:
        ds = load_dataset("MBZUAI/VideoMathQA", "multi_binary")
        raw_train, raw_test = load_videomathqa(ds)
    """
    split     = hf_dataset["test"].train_test_split(test_size=test_size, seed=seed)
    raw_train = split["train"]
    raw_test  = split["test"]
    print(f"  VideoMathQA split → train: {len(raw_train):,}  test: {len(raw_test):,}  (seed={seed})")
    return raw_train, raw_test


VIDEOMATHQA_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "You will be shown a video and a multiple-choice question. "
    "Think step by step, then end your response with 'Answer: A' or 'Answer: B'."
)


class VideoMathQADataset(Dataset):
    """
    Dataset class for MBZUAI/VideoMathQA (multi_binary split).

    Each sample is a multiple-choice video QA item with exactly 2 options (A/B).

    - is_test=False : assistant turn = step-by-step reasoning + final answer
    - is_test=True  : no assistant turn, generation prompt added instead

    Args:
        dataset       : HuggingFace Dataset object (train or test slice)
        processor     : AutoProcessor for Qwen3-VL
        video_base    : Path to folder containing video files
        max_length    : Max token length for truncation
        fps           : Frames per second to sample from video
        max_frames    : Maximum number of frames to extract
        is_test       : If True, no answer/reasoning in input (eval mode)
        video_ext     : Video file extension (default: .mp4)
    """

    def __init__(
        self,
        dataset,
        processor,
        video_base,
        max_length  = 4096,
        fps         = 1.0,
        max_frames  = 8,
        is_test     = False,
        video_ext   = ".mp4",
    ):
        self.dataset    = dataset
        self.processor  = processor
        self.video_base = video_base
        self.max_length = max_length
        self.fps        = fps
        self.max_frames = max_frames
        self.is_test    = is_test
        self.video_ext  = video_ext

    def __len__(self):
        return len(self.dataset)

    def _format_options(self, options):
        return "\n".join(options)

    def _build_question_text(self, question, options):
        return (
            f"{question}\n\n"
            f"Options:\n{self._format_options(options)}\n\n"
            f"Think step by step, then end with 'Answer: A' or 'Answer: B'."
        )

    def _format_steps(self, steps_raw):
        if isinstance(steps_raw, str):
            steps = json.loads(steps_raw)
        else:
            steps = steps_raw
        return "\n".join(f"Step {k}: {v}" for k, v in steps.items())

    def __getitem__(self, idx):
        item = self.dataset[idx]

        video_path    = os.path.join(self.video_base, item["videoID"] + self.video_ext)
        question_text = self._build_question_text(item["question"], item["options"])

        messages = [
            {"role": "system", "content": VIDEOMATHQA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type":                 "video",
                        "video":                video_path,
                        "fps":                  self.fps,
                        "max_frames":           self.max_frames,
                        "video_reader_backend": "decord",   # avoids swscaler/PyAV crashes
                    },
                    {"type": "text", "text": question_text},
                ],
            },
        ]

        if not self.is_test:
            reasoning  = self._format_steps(item["steps"])
            answer_str = f"{reasoning}\n\nAnswer: {item['answer']}"
            messages.append({"role": "assistant", "content": answer_str})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=self.is_test)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)

        # fps comes back as list [x.xx] from process_vision_info — drop it
        video_kwargs.pop("fps", None)

        inputs = self.processor(
            text           = [text],
            images         = image_inputs,
            videos         = video_inputs,
            padding        = "max_length",
            truncation     = not self.is_test,
            max_length     = self.max_length,
            return_tensors = "pt",
            **video_kwargs,
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Metadata for evaluator
        inputs["answer"]      = item["answer"]
        inputs["question_id"] = item["question_id"]
        inputs["mbinary_id"]  = item["mbinary_id"]
        inputs["category"]    = item["category"]

        return inputs
    
    



LEVELGAUGE_SYSTEM_PROMPT = (
    "You are a video understanding assistant. "
    "You will be shown a video and a multiple-choice question with options A, B, C, or D. "
    "Think step by step, then end your response with '<answer>A', '<answer>B', "
    "'<answer>C', or '<answer>D'."
)

LETTER_MAP = ["A", "B", "C", "D"]


def load_levelgauge(hf_dataset, test_size=0.2, seed=42):
    """
    Split Video-LevelGauge into train and test subsets.

    Usage:
        ds = load_dataset("Cola-any/Video-LevelGauge")
        raw_train, raw_test = load_levelgauge(ds)
    """
    # Use whichever split is available
    split_name = "train" if "train" in hf_dataset else list(hf_dataset.keys())[0]
    split      = hf_dataset[split_name].train_test_split(test_size=test_size, seed=seed)
    raw_train  = split["train"]
    raw_test   = split["test"]
    print(f"  Video-LevelGauge split → train: {len(raw_train):,}  "
          f"test: {len(raw_test):,}  (seed={seed})")
    return raw_train, raw_test


class VideoLevelGaugeDataset(Dataset):
    """
    Dataset class for Cola-any/Video-LevelGauge.

    Each sample is a multiple-choice video QA item with 4 candidates (A-D).
    The gt_answer is matched against candidates to find the correct letter.

    - is_test=False : assistant turn = reasoning + '<answer>X'
    - is_test=True  : no assistant turn, generation prompt added instead

    Args:
        dataset       : HuggingFace Dataset object (train or test slice)
        processor     : AutoProcessor for Qwen3-VL
        video_base    : Path to folder containing video files
        max_length    : Max token length for truncation
        fps           : Frames per second to sample from video
        max_frames    : Maximum number of frames to extract
        is_test       : If True, no answer in input (eval mode)
        video_ext     : Video file extension (default: .mp4, already in video_name)
    """

    def __init__(
        self,
        dataset,
        processor,
        video_base,
        max_length  = 4096,
        fps         = 1.0,
        max_frames  = 8,
        is_test     = False,
    ):
        self.dataset    = dataset
        self.processor  = processor
        self.video_base = video_base
        self.max_length = max_length
        self.fps        = fps
        self.max_frames = max_frames
        self.is_test    = is_test

    def __len__(self):
        return len(self.dataset)

    def _format_candidates(self, candidates):
        """
        Format list of candidates as lettered options.
        ['Neck.', 'Chest.', 'Chin.', 'Nose.']
        →
        'A. Neck.\nB. Chest.\nC. Chin.\nD. Nose.'
        """
        return "\n".join(
            f"{LETTER_MAP[i]}. {c}" for i, c in enumerate(candidates)
        )

    def _get_answer_letter(self, gt_answer, candidates):
        """
        Find the letter (A/B/C/D) corresponding to gt_answer by matching
        against the candidates list. Falls back to 'A' if not found.
        """
        for i, candidate in enumerate(candidates):
            if candidate.strip().lower() == gt_answer.strip().lower():
                return LETTER_MAP[i]
        # Partial match fallback
        for i, candidate in enumerate(candidates):
            if gt_answer.strip().lower() in candidate.strip().lower():
                return LETTER_MAP[i]
        return LETTER_MAP[0]   # fallback

    def _build_question_text(self, question, candidates):
        return (
            f"{question}\n\n"
            f"Options:\n{self._format_candidates(candidates)}\n\n"
            f"Think step by step, then end with '<answer>A', '<answer>B', "
            f"'<answer>C', or '<answer>D'."
        )

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # video_name already includes the extension (e.g. '21q-lDikdBg.mp4')
        video_path    = os.path.join(self.video_base, item["video_name"])
        candidates    = item["candidates"]
        answer_letter = self._get_answer_letter(item["gt_answer"], candidates)
        question_text = self._build_question_text(item["question"], candidates)

        # ── Build conversation ────────────────────────────────────────────────
        messages = [
            {"role": "system", "content": LEVELGAUGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type":       "video",
                        "video":      video_path,
                        "fps":        self.fps,
                        "max_frames": self.max_frames,
                    },
                    {"type": "text", "text": question_text},
                ],
            },
        ]

        if not self.is_test:
            # Assistant turn: answer with the correct letter in <answer> format
            messages.append({
                "role":    "assistant",
                "content": f"<answer>{answer_letter}",
            })

        # ── Tokenise ──────────────────────────────────────────────────────────
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=self.is_test)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)

        # fps comes back as list [x.xx] from process_vision_info — drop it
        video_kwargs.pop("fps", None)

        inputs = self.processor(
            text           = [text],
            images         = image_inputs,
            videos         = video_inputs,
            padding        = "max_length",
            truncation     = not self.is_test,
            max_length     = self.max_length,
            return_tensors = "pt",
            **video_kwargs,
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Metadata for evaluator
        inputs["answer"]        = answer_letter           # e.g. "B"
        inputs["gt_answer"]     = item["gt_answer"]       # e.g. "Chest."
        inputs["question_id"]   = item["question_id"]
        inputs["question_type"] = item["question_type"]
        inputs["video_category"]= item["video_category"]

        return inputs
    
    
"""
Image Classification Dataset Classes for ViT (google/vit-base-patch16-224)

Datasets:
    ImagenetteDataset       — torchvision Imagenette (train/test split from train)
    BeansDataset            — beans (train/validation/test)
    OxfordPetsDataset       — timm/oxford-iiit-pet (train/test)
    ChestXRayDataset        — hf-vision/chest-xray-pneumonia (train/validation/test)
    RVLCDIPDataset          — vaclavpechtor/rvl_cdip-small-200 (train/val as test)
    AIWD6Dataset            — Kondapally/AIWD6 (train/validation/test)

All datasets return:
    pixel_values : torch.Tensor  [3, 224, 224]  (processed by ViT feature extractor)
    labels       : int           class index

Note: ViT produces fixed-length sequences (197 tokens = 196 patches + 1 CLS).
      Right padding is used as convention for bidirectional models.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# =============================================================================
#  BASE CLASS
# =============================================================================

class BaseImageCLSDataset(Dataset):
    """
    Base class for all image classification datasets.
    Subclasses only need to implement __len__ and _get_image_label.
    """

    def __init__(self, processor):
        self.processor = processor

    def __len__(self):
        raise NotImplementedError

    def _get_image_label(self, idx):
        """Return (PIL.Image, int label)"""
        raise NotImplementedError

    def __getitem__(self, idx):
        image, label = self._get_image_label(idx)

        # Ensure RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 224, 224]

        return {
            "pixel_values": pixel_values,
            "labels":       torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
#  1. IMAGENETTE
# =============================================================================

class ImagenetteDataset(BaseImageCLSDataset):
    """
    Imagenette from torchvision.
    No official test split — we split the train set 80/20.

    Usage:
        from torchvision.datasets import Imagenette
        raw = Imagenette(root="./data", split="train", size="320px", download=True)
        raw_train, raw_test = load_imagenette(raw)
        train_ds = ImagenetteDataset(raw_train, processor)
        test_ds  = ImagenetteDataset(raw_test,  processor)
    """

    def __init__(self, raw_dataset, processor):
        super().__init__(processor)
        self.dataset = raw_dataset

    def __len__(self):
        return len(self.dataset)

    def _get_image_label(self, idx):
        image, label = self.dataset[idx]
        # torchvision returns PIL image directly (no transform applied if not set)
        if not isinstance(image, Image.Image):
            # If tensor (transform was applied), convert back
            import torchvision.transforms.functional as F
            if hasattr(image, 'numpy'):
                image = Image.fromarray(
                    (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        return image, label


def load_imagenette(raw_dataset, test_size=0.2, seed=42):
    """
    Split Imagenette train set into train/test subsets.

    Usage:
        raw = Imagenette(root="./data", split="train", size="320px", download=True)
        raw_train, raw_test = load_imagenette(raw)
    """
    import random
    random.seed(seed)
    indices   = list(range(len(raw_dataset)))
    random.shuffle(indices)
    split_at  = int(len(indices) * (1 - test_size))
    train_idx = indices[:split_at]
    test_idx  = indices[split_at:]

    from torch.utils.data import Subset
    raw_train = Subset(raw_dataset, train_idx)
    raw_test  = Subset(raw_dataset, test_idx)
    print(f"  Imagenette split → train: {len(raw_train):,}  test: {len(raw_test):,}  (seed={seed})")
    return raw_train, raw_test


# =============================================================================
#  2. BEANS
# =============================================================================

class BeansDataset(BaseImageCLSDataset):
    """
    Beans dataset from HuggingFace.
    Features: image_file_path, image, labels
    3 classes: angular_leaf_spot, bean_rust, healthy

    Usage:
        ds = load_dataset("beans")
        train_ds = BeansDataset(ds["train"],      processor)
        val_ds   = BeansDataset(ds["validation"], processor)
        test_ds  = BeansDataset(ds["test"],       processor)
    """

    def __init__(self, hf_split, processor):
        super().__init__(processor)
        self.dataset = hf_split

    def __len__(self):
        return len(self.dataset)

    def _get_image_label(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["labels"]
        return image, label


# =============================================================================
#  3. OXFORD-IIIT PETS
# =============================================================================

class OxfordPetsDataset(BaseImageCLSDataset):
    """
    Oxford-IIIT Pet dataset from timm/oxford-iiit-pet.
    Features: image, label, image_id, label_cat_dog
    37 classes (pet breeds).

    Usage:
        ds = load_dataset("timm/oxford-iiit-pet")
        train_ds = OxfordPetsDataset(ds["train"], processor)
        test_ds  = OxfordPetsDataset(ds["test"],  processor)
    """

    def __init__(self, hf_split, processor):
        super().__init__(processor)
        self.dataset = hf_split

    def __len__(self):
        return len(self.dataset)

    def _get_image_label(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        return image, label


# =============================================================================
#  4. CHEST X-RAY PNEUMONIA
# =============================================================================

class ChestXRayDataset(BaseImageCLSDataset):
    """
    Chest X-Ray Pneumonia dataset from hf-vision/chest-xray-pneumonia.
    Features: image, label
    2 classes: NORMAL, PNEUMONIA

    Usage:
        ds = load_dataset("hf-vision/chest-xray-pneumonia")
        train_ds = ChestXRayDataset(ds["train"],      processor)
        val_ds   = ChestXRayDataset(ds["validation"], processor)
        test_ds  = ChestXRayDataset(ds["test"],       processor)
    """

    def __init__(self, hf_split, processor):
        super().__init__(processor)
        self.dataset = hf_split

    def __len__(self):
        return len(self.dataset)

    def _get_image_label(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        return image, label


# =============================================================================
#  5. RVL-CDIP SMALL
# =============================================================================

class RVLCDIPDataset(BaseImageCLSDataset):
    """
    RVL-CDIP Small dataset from vaclavpechtor/rvl_cdip-small-200.
    Features: image, label
    16 document classes.
    No test split — use validation as test.

    Usage:
        ds = load_dataset("vaclavpechtor/rvl_cdip-small-200")
        train_ds = RVLCDIPDataset(ds["train"],      processor)
        test_ds  = RVLCDIPDataset(ds["validation"], processor)  # val as test
    """

    def __init__(self, hf_split, processor):
        super().__init__(processor)
        self.dataset = hf_split

    def __len__(self):
        return len(self.dataset)

    def _get_image_label(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        return image, label


# =============================================================================
#  6. AIWD6
# =============================================================================

class AIWD6Dataset(BaseImageCLSDataset):
    """
    AIWD6 dataset from Kondapally/AIWD6.
    Features: image, label

    Usage:
        ds = load_dataset("Kondapally/AIWD6")
        train_ds = AIWD6Dataset(ds["train"],      processor)
        test_ds  = AIWD6Dataset(ds["test"],       processor)
        # or use validation as test if needed:
        test_ds  = AIWD6Dataset(ds["validation"], processor)
    """

    def __init__(self, hf_split, processor):
        super().__init__(processor)
        self.dataset = hf_split

    def __len__(self):
        return len(self.dataset)

    def _get_image_label(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        return image, label
    
    
    
    
    
# =============================================================================
#  ADD THESE TO YOUR dataset.py
#  (alongside the existing OxfordPetsDataset, ChestXRayDataset, etc.)
# =============================================================================

# Make sure these are at the top of dataset.py if not already there:
#   import numpy as np
#   from PIL import Image
#   from torch.utils.data import Dataset
#   import torch


# =============================================================================
#  CIFAR-10
# =============================================================================

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 from uoft-cs/cifar10.
    Features: img (image), label
    10 classes: airplane, automobile, bird, cat, deer,
                dog, frog, horse, ship, truck

    Usage:
        ds = load_dataset("uoft-cs/cifar10")
        train_ds = CIFAR10Dataset(ds["train"], processor)
        test_ds  = CIFAR10Dataset(ds["test"],  processor)
    """

    def __init__(self, hf_split, processor):
        self.dataset   = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item["img"]     # key is 'img' not 'image' for cifar10
        label = item["label"]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
#  SUN-397
# =============================================================================

class SUN397Dataset(Dataset):
    """
    SUN-397 scene classification from tanganke/sun397.
    Features: image, label
    397 scene categories.

    Usage:
        ds = load_dataset("tanganke/sun397")
        train_ds = SUN397Dataset(ds["train"], processor)
        test_ds  = SUN397Dataset(ds["test"],  processor)
    """

    def __init__(self, hf_split, processor):
        self.dataset   = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
#  EUROSAT
# =============================================================================

class EuroSATDataset(Dataset):
    """
    EuroSAT satellite image classification from tanganke/eurosat.
    Features: image, label
    10 land use classes (e.g. annual crop, forest, highway, etc.)

    Usage:
        ds = load_dataset("tanganke/eurosat")
        train_ds = EuroSATDataset(ds["train"], processor)
        test_ds  = EuroSATDataset(ds["test"],  processor)
    """

    def __init__(self, hf_split, processor):
        self.dataset   = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
#  AMERICAN SIGN LANGUAGE MNIST
# =============================================================================

class ASLMNISTDataset(Dataset):
    """
    American Sign Language MNIST from rrrajjj/American-Sign-Language-MNIST.
    Features: image, label
    275 classes (ASL hand signs).
    Grayscale images — converted to RGB for ViT.

    Usage:
        ds = load_dataset("rrrajjj/American-Sign-Language-MNIST")
        train_ds = ASLMNISTDataset(ds["train"], processor)
        test_ds  = ASLMNISTDataset(ds["test"],  processor)
    """

    def __init__(self, hf_split, processor):
        self.dataset   = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # ASL MNIST is grayscale — convert to RGB for ViT
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
#  TOMATO LEAVES
# =============================================================================

class TomatoLeavesDataset(Dataset):
    """
    Tomato Leaves Disease Classification from Vlone571/tomato-leaves-dataset.
    Features: image, label
    8 disease classes (e.g. Tomato___Bacterial_spot, Tomato___healthy, etc.)

    Usage:
        ds = load_dataset("Vlone571/tomato-leaves-dataset")
        train_ds = TomatoLeavesDataset(ds["train"],      processor)
        test_ds  = TomatoLeavesDataset(ds["test"],       processor)
        # or if no test split:
        test_ds  = TomatoLeavesDataset(ds["validation"], processor)
    """

    def __init__(self, hf_split, processor):
        self.dataset   = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }