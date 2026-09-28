from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import tyro


@dataclass
class Args:
    """Arguments for launching the vLLM OpenAI-compatible server."""

    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    download_dir: str | None = None
    log_file: str | None = None
    served_model_name: str | None = None

    trust_remote_code: bool = True
    max_num_batched_tokens: int | None = 32768
def _build_command(args: Args) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.download_dir:
        cmd.extend(["--download-dir", args.download_dir])
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    return cmd


def main(args: Args) -> None:
    try:
        import vllm  # noqa: F401  # pylint: disable=unused-import
    except Exception as exc:  # pragma: no cover - optional dependency guard
        try:
            torch_version = version("torch")
        except PackageNotFoundError:
            torch_version = "not installed"
        try:
            vllm_version = version("vllm")
        except PackageNotFoundError:
            vllm_version = "not installed"
        raise RuntimeError(
            "Unable to import vLLM "
            f"(vllm={vllm_version}, torch={torch_version}). "
            "vLLM contains compiled CUDA extensions, so changing torch alone does not "
            "change its CUDA ABI. Recreate/sync the complete environment instead. "
            "For Molmo2 in this repository run `uv sync --extra molmo`; the locked "
            "combination is vllm 0.15.0 with torch 2.9.1."
        ) from exc

    cmd = _build_command(args)
    env = os.environ.copy()

    log_handle: int | None = None
    stdout = stderr = None
    if args.log_file:
        log_handle = os.open(args.log_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        stdout = stderr = log_handle

    process = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)

    def _forward_signal(signum: int, _frame: object | None) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    try:
        rc = process.wait()
    finally:
        if log_handle is not None:
            os.close(log_handle)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main(tyro.cli(Args))
