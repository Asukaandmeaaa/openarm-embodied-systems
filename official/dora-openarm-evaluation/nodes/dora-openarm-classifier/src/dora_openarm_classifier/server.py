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

"""Classification server.

Standalone server that loads model once and serves classification requests.
Clients can send images via HTTP POST and get back success scores.

Usage:
    dora-openarm-classifier-server --port 8000

API:
    POST /classify
    Content-Type: multipart/form-data
      question: yes/no success question
      image: JPEG or PNG image
        Content-Type: image/jpeg or image/png

    Response (JSON):
    {
        "p_yes": 0.85,
        "p_no": 0.12,
        "score": 0.876,  # normalized p_yes
    }
"""

import argparse
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Annotated
import uvicorn

from .topreward import TOPRewardScorer

app = FastAPI()

scorer = None
args = None


@app.get("/health", response_class=JSONResponse)
def _health():
    return {"status": "healthy"}


@app.post("/classify", response_class=JSONResponse)
async def _classify(
    question: Annotated[str, Form()], image: Annotated[UploadFile, File()]
):
    p_yes, p_no, score = scorer.score(Image.open(image.file).convert("RGB"), question)
    return {
        "p_yes": float(p_yes),
        "p_no": float(p_no),
        "score": float(score),
    }


def main():
    """Run TOPReward based classification server."""
    parser = argparse.ArgumentParser(description="Classification server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID (default: Qwen/Qwen3-VL-4B-Instruct)",
    )

    global args
    args = parser.parse_args()

    print("Loading model...", flush=True)
    global scorer
    if args.model_id:
        scorer = TOPRewardScorer(args.model_id)
    else:
        scorer = TOPRewardScorer()
    print("Ready!", flush=True)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)

    server.run()


if __name__ == "__main__":
    main()
