# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen-VL TOPReward scorer.

https://topreward.github.io/webpage/

Asks a yes/no question about an image and reads the probability the model
assigns to "Yes" from the next-token logits. The normalized
P_yes = P(yes) / (P(yes) + P(no)) is the success score -- no labels, no training.
"""

from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
YES_WORDS = ["Yes", "yes", "YES", " Yes", " yes"]
NO_WORDS = ["No", "no", "NO", " No", " no"]


def _load_model(model_id):
    """Pick the right model class from the id.

    Use Instruct (not Thinking)
    variants: we read the FIRST generated token, but Thinking models emit
    reasoning first, so the first token would not be Yes/No.
    """
    low = model_id.lower()
    if "qwen3-vl" in low and "moe" in low:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
    elif "qwen3-vl" in low:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    else:  # qwen2.5-vl and friends
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    return Cls.from_pretrained(model_id, torch_dtype="auto", device_map="auto").eval()


class TOPRewardScorer:
    """Single-frame Qwen-VL success scorer."""

    def __init__(self, model_id=MODEL_ID):
        """Load the model and processor once."""
        print(f"[load] {model_id} ...", flush=True)
        self.model = _load_model(model_id)
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.tok = self.proc.tokenizer
        print(
            f"[load] done. device={self.model.device} dtype={self.model.dtype}\n",
            flush=True,
        )

    def _mass(self, probs, words):
        s = 0.0
        for w in words:
            ids = self.tok.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                s += float(probs[ids[0]].item())
        return s

    def next_token_probs(self, image, question):
        """Next-token probability distribution for (image, question).

        ``image`` may be a file path or a PIL image.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        return torch.softmax(logits.float(), dim=-1)

    def score(self, image, question):
        """Return (raw P_yes, raw P_no, normalized P_yes = TOPReward)."""
        probs = self.next_token_probs(image, question + "\n\nAnswer Yes or No.")
        p_yes = self._mass(probs, YES_WORDS)
        p_no = self._mass(probs, NO_WORDS)
        norm = p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.0
        return p_yes, p_no, norm
