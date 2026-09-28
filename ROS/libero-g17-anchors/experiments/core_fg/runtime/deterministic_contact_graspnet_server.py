"""Request-deterministic wrapper for the CaP-X ContactGraspNet service.

This wrapper does not change the grasp model or its inputs. It derives a seed
from the complete request payload and resets the service-local RNGs immediately
before inference. Consequently, byte-identical requests receive the same random
point-cloud subsample regardless of request order or earlier trials.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np
import torch
import tyro

from capx.serving import launch_contact_graspnet_server as _server


_original_do_plan = _server._do_plan


def _request_payload(req: Any) -> dict[str, Any]:
    if hasattr(req, "model_dump"):
        return req.model_dump(mode="json")
    if hasattr(req, "dict"):
        return req.dict()
    raise TypeError(f"Unsupported ContactGraspNet request type: {type(req)!r}")


def request_sha256_and_seed(req: Any) -> tuple[str, int]:
    canonical = json.dumps(
        _request_payload(req),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    # NumPy's legacy global RNG accepts a 32-bit seed.
    seed = int(digest[:8], 16)
    return digest, seed


def _seed_service_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _array_sha256(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _do_plan_deterministic(req: Any):
    request_sha256, seed = request_sha256_and_seed(req)
    _seed_service_rngs(seed)
    response = _original_do_plan(req)
    response_sha256 = hashlib.sha256(
        json.dumps(
            {
                "grasps": _array_sha256(response.grasps_base64),
                "scores": _array_sha256(response.scores_base64),
                "contact_pts": _array_sha256(response.contact_pts_base64),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    print(
        "[capx deterministic graspnet] "
        f"request={request_sha256} seed={seed} response={response_sha256}",
        flush=True,
    )
    return response


def main(device: str = "cuda", port: int = 8115, host: str = "127.0.0.1"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    _server._do_plan = _do_plan_deterministic
    return _server.main(device=device, port=port, host=host)


if __name__ == "__main__":
    tyro.cli(main)
