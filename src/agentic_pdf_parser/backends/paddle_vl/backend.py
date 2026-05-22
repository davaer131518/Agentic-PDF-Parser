"""PaddleVL backend: inference + normalization in a single parse_page() call.

This module satisfies the single-contract rule:
  ``parse_page()`` owns both model inference AND normalization.
  It returns ``BackendPageResult(page=<already normalized Page>, raw=...)``.
  The orchestrator never normalizes.

Public backend name: ``paddle_vl``
Underlying pipeline: ``PaddleOCR-VL-1.5`` (PaddleOCR 3.x + PP-DocLayoutV3)

Architecture
------------
The backend uses a hybrid inference approach:

* **PP-DocLayoutV3** (layout detection, reading order, bounding boxes) runs via
  the PaddleOCR Python package as before.
* **VLM recognition** (OCR text, table HTML, formula LaTeX per block) is
  offloaded to ``llama-server`` (``llama-server.exe`` on Windows) serving
  ``PaddleOCR-VL-1.5.gguf``.
  ``PaddleOCRVL`` natively supports ``vl_rec_backend="llama-cpp-server"``.

``llama-server`` is started as a background subprocess in ``load()`` and
terminated in ``unload()``.  All ``parse_page()`` calls happen between these
two lifecycle methods with the server already running.

Raw output format
-----------------
The ``raw`` field of :class:`BackendPageResult` is unchanged from the original
implementation — a plain JSON-serializable dict built from the
``PaddleOCRVLResult`` object::

    {
        "page_index": int,
        "width": int,
        "height": int,
        "parsing_res_list": [
            {
                "block_label": str,
                "block_content": str,
                "block_bbox": [x1, y1, x2, y2],
                "block_id": int,
            },
            ...
        ],
    }
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import urlopen

from ...backends.base import BackendPageResult, PageInput, ParserBackend
from ...config import ParseConfig
from ...utils.logging import get_logger
from . import device as _device_mod
from . import normalizer as _normalizer

if TYPE_CHECKING:
    pass  # PaddleOCRVL imported lazily in load()

logger = get_logger(__name__)

_HEALTH_TIMEOUT_S = 60   # max seconds to wait for llama-server to become ready
_HEALTH_POLL_S = 1.0     # polling interval


def _resolve_server_bin(llama_cpp_dir: Path) -> Path:
    """Return the path to the llama-server binary, trying two layouts.

    Supports:
    - Windows flat release layout: ``<dir>/llama-server.exe``
    - Non-Windows flat layout:     ``<dir>/llama-server``
    - Homebrew / packaged layout:  ``<dir>/bin/llama-server``
      (e.g. ``/opt/homebrew/bin/llama-server``)

    Returns the first path that exists.  If neither exists the flat path is
    returned so the caller's ``FileNotFoundError`` message is informative.
    """
    binary = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    direct = llama_cpp_dir / binary
    if direct.exists():
        return direct
    via_bin = llama_cpp_dir / "bin" / binary
    if via_bin.exists():
        return via_bin
    return direct  # return for a meaningful FileNotFoundError


class PaddleVLBackend:
    """Parser backend backed by ``PaddleOCR-VL-1.5`` via PaddleX + llama-server.

    PP-DocLayoutV3 is an internal component of this pipeline — it is never
    exposed as a public backend name.  The public name is ``paddle_vl``.

    Call sequence expected by the orchestrator::

        backend.load()
        try:
            for page in pages:
                result = backend.parse_page(page_input)
        finally:
            backend.unload()
    """

    name: str = "paddle_vl"
    version: str | None = None
    model_id: str | None = "PaddleOCR-VL-1.5"
    resolved_device: str = "cpu"

    def __init__(self, cfg: ParseConfig) -> None:
        self._cfg = cfg
        self.resolved_device = _device_mod.resolve_device(cfg.device)
        self._pipeline: Any = None
        self._llama_proc: subprocess.Popen[str] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Start llama-server and initialise the PaddleOCR-VL pipeline.

        Steps:
        1. Validate that all required llama.cpp binary and GGUF paths exist.
        2. Launch ``llama-server`` (``llama-server.exe`` on Windows) as a background process.
        3. Poll the ``/health`` endpoint until it reports ready.
        4. Import ``paddleocr`` lazily and create the ``PaddleOCRVL`` pipeline
           pointing at the running server.

        Raises
        ------
        FileNotFoundError
            If any required binary or model file is missing.
        RuntimeError
            If ``paddlepaddle`` is not installed, or the server fails to start.
        TimeoutError
            If llama-server does not become ready within ``_HEALTH_TIMEOUT_S``.
        """
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        p = self._cfg.paddle_vl
        server_bin = _resolve_server_bin(p.llama_cpp_dir)
        gguf_path = p.gguf_model_path
        mmproj_path = p.mmproj_path

        for label, path in [
            (server_bin.name, server_bin),
            ("gguf_model_path", gguf_path),
            ("mmproj_path", mmproj_path),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"PaddleVLBackend: {label} not found at {path}. "
                    "Check PaddleVLConfig paths in your YAML config."
                )

        # Resolve GPU offload layers for llama-server
        n_gpu_layers = _device_mod.resolve_n_gpu_layers(self._cfg.device)

        logger.info(
            "Starting llama-server for PaddleOCR-VL on port %d (n_gpu_layers=%d)",
            p.server_port, n_gpu_layers,
        )
        cmd = [
            str(server_bin),
            "-m", str(gguf_path),
            "--mmproj", str(mmproj_path),
            "--port", str(p.server_port),
            "--host", "127.0.0.1",
            "--temp", "0",
            "-ngl", str(n_gpu_layers),
        ]
        self._llama_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        self._wait_for_server(p.server_port)

        _install_torch_stub()

        try:
            from paddleocr import PaddleOCRVL  # noqa: PLC0415
        except ImportError as exc:
            self._terminate_server()
            raise RuntimeError(
                f"Failed to import PaddleOCRVL: {exc}. "
                "Ensure paddleocr and paddlepaddle (or paddlepaddle-gpu) are installed: "
                "pip install 'agentic-pdf-parser[paddle]'"
            ) from exc

        paddle_cfg = p
        server_url = f"http://127.0.0.1:{p.server_port}/v1"

        logger.info(
            "Loading PaddleOCR-VL pipeline on device=%s "
            "(vl_rec_backend=llama-cpp-server, url=%s, "
            "orient_classify=%s, unwarping=%s)",
            self.resolved_device,
            server_url,
            paddle_cfg.use_doc_orientation_classify,
            paddle_cfg.use_doc_unwarping,
        )

        try:
            self._pipeline = PaddleOCRVL(
                device=self.resolved_device,
                vl_rec_backend="llama-cpp-server",
                vl_rec_server_url=server_url,
                use_doc_orientation_classify=paddle_cfg.use_doc_orientation_classify,
                use_doc_unwarping=paddle_cfg.use_doc_unwarping,
            )
        except Exception as exc:
            self._terminate_server()
            raise RuntimeError(
                f"Failed to create PaddleOCR-VL pipeline: {exc}. "
                "Ensure paddlepaddle (or paddlepaddle-gpu) is installed."
            ) from exc

        self.version = "PaddleOCR-VL-1.5"
        logger.info(
            "PaddleOCR-VL pipeline ready (device=%s, llama-server port=%d)",
            self.resolved_device, p.server_port,
        )

    def unload(self) -> None:
        """Shut down llama-server and release pipeline resources."""
        if self._pipeline is not None:
            try:
                self._pipeline.close()
            except Exception:
                pass
            self._pipeline = None
        self._terminate_server()

    # ------------------------------------------------------------------
    # Single-contract entry point
    # ------------------------------------------------------------------

    def parse_page(self, page_input: PageInput) -> BackendPageResult:
        """Run inference + normalization on a single page image.

        Steps (all internal — orchestrator never normalizes):
        1. Call ``PaddleOCRVL.predict()`` on the page image path.
           PP-DocLayoutV3 detects layout blocks; llama-server recognizes
           each block's content (text, table HTML, formula LaTeX).
        2. Serialize the result to a JSON-safe dict via ``_result_to_raw()``.
        3. Call ``normalizer.raw_result_to_page()`` to produce a canonical Page.
        4. Return ``BackendPageResult(page=<normalized>, raw=<dict>)``.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "PaddleVLBackend.parse_page() called before load(). "
                "Call backend.load() first."
            )

        results = self._pipeline.predict(str(page_input.temp_image_path))
        if not results:
            raise RuntimeError(
                f"PaddleOCR-VL returned an empty result for page "
                f"{page_input.page_number}."
            )

        result = results[0]
        raw = _result_to_raw(result)

        page = _normalizer.raw_result_to_page(
            raw=raw,
            page_input=page_input,
            backend_name=self.name,
            backend_version=self.version,
        )
        return BackendPageResult(page=page, raw=raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_server(self, port: int) -> None:
        """Poll the llama-server health endpoint until it responds or times out."""
        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            # Check if the process died early
            if self._llama_proc and self._llama_proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server process exited unexpectedly "
                    f"(exit code {self._llama_proc.returncode})."
                )
            try:
                with urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info("llama-server ready on port %d", port)
                        return
            except (URLError, OSError):
                pass
            time.sleep(_HEALTH_POLL_S)
        self._terminate_server()
        raise TimeoutError(
            f"llama-server did not become ready within {_HEALTH_TIMEOUT_S}s "
            f"(port {port})."
        )

    def _terminate_server(self) -> None:
        """Terminate the llama-server subprocess if it is running."""
        if self._llama_proc is not None:
            try:
                # On Windows, terminate() sends CTRL_BREAK which llama-server
                # may ignore. Use kill() directly to ensure the process exits.
                self._llama_proc.kill()
                self._llama_proc.wait(timeout=10)
            except Exception:
                pass
            self._llama_proc = None
            logger.info("llama-server terminated.")


# ---------------------------------------------------------------------------
# Windows CUDA DLL registration + torch stub
# ---------------------------------------------------------------------------


def _install_torch_stub() -> None:
    """Ensure ``import torch`` never raises an OSError in this process.

    ``modelscope`` (a ``paddlex`` / ``paddleocr`` transitive dependency)
    imports ``torch`` unconditionally at module load time — only to call
    ``torch.distributed.is_initialized()`` and ``torch.distributed.get_rank()``
    for distributed-training bookkeeping.  We never use torch for inference
    (PaddleVL VLM inference runs via llama-server).

    On some Windows environments the torch DLL loader raises an OSError even
    for CPU-only wheels (``shm.dll`` or ``cudnn_cnn64_9.dll`` missing
    dependencies).  When that happens we register a minimal stub that satisfies
    every attribute modelscope accesses, so the import chain completes without
    loading any native DLLs.

    If torch can already be imported cleanly, this function is a no-op.
    """
    import sys  # noqa: PLC0415
    from types import ModuleType  # noqa: PLC0415

    if "torch" in sys.modules:
        return  # already imported or stubbed — nothing to do

    try:
        import torch  # noqa: PLC0415, F401
        return  # imported cleanly — nothing to do
    except OSError:
        pass  # DLL load failure — fall through to stub

    logger.debug(
        "torch DLL load failed; installing stub meta-path finder to satisfy "
        "modelscope/paddleocr import chain (no torch inference is performed)."
    )

    import importlib.abc       # noqa: PLC0415
    import importlib.machinery  # noqa: PLC0415

    # A metaclass-based stub so that any torch attribute (e.g. torch.Tensor,
    # torch.nn.Module) looks like a real class to issubclass() / isinstance().
    class _StubMeta(type):
        """Metaclass for stub types — supports chained attribute access on classes."""

        def __getattr__(cls, name: str) -> type:
            # Return a new stub class for any attribute access on a stub class.
            return _StubMeta(f"{cls.__name__}.{name}", (_StubBase,), {})

        def __call__(cls, *args: object, **kwargs: object) -> None:
            return None

        def __bool__(cls) -> bool:
            return False

        def __instancecheck__(cls, instance: object) -> bool:
            return False

        def __subclasscheck__(cls, subclass: object) -> bool:
            return False

    class _StubBase(metaclass=_StubMeta):
        """Base stub type — a real Python class usable as issubclass() argument."""
        pass

    class _TorchStubModule(ModuleType):
        """A stub module that returns real stub types for any attribute access.

        Handles all torch usage patterns:
          - ``import torch.nn``                    → MetaPathFinder creates this
          - ``torch.Tensor``                       → returns _StubBase subclass (a type)
          - ``issubclass(x, torch.Tensor)``        → works (it's a real type)
          - ``torch.distributed.is_available()``  → pre-set lambda on the module
          - ``isinstance(x, torch.nn.Module)``    → works (__instancecheck__ → False)
        """

        def __getattr__(self, name: str) -> object:
            full = f"{self.__name__}.{name}"
            # If the submodule has been imported via MetaPathFinder, return it.
            if full in sys.modules:
                return sys.modules[full]
            # Otherwise return a stub class (a real type, so issubclass works).
            stub_cls = _StubMeta(full, (_StubBase,), {})
            object.__setattr__(self, name, stub_cls)
            return stub_cls

    class _TorchStubLoader(importlib.abc.Loader):
        def create_module(
            self, spec: importlib.machinery.ModuleSpec
        ) -> "_TorchStubModule":
            mod = _TorchStubModule(spec.name)
            mod.__spec__ = spec
            mod.__path__ = []  # type: ignore[assignment]
            mod.__version__ = "0.0.0+stub"  # type: ignore[attr-defined]
            # Pre-set torch.distributed helpers so modelscope gets proper booleans.
            if spec.name == "torch.distributed":
                object.__setattr__(mod, "is_available", lambda: False)
                object.__setattr__(mod, "is_initialized", lambda: False)
                object.__setattr__(mod, "get_rank", lambda: 0)
                object.__setattr__(mod, "get_world_size", lambda: 1)
            return mod

        def exec_module(self, module: ModuleType) -> None:
            pass  # fully configured in create_module

    class _TorchStubFinder(importlib.abc.MetaPathFinder):
        """Intercept every 'torch' and 'torch.*' import and return a stub."""

        _loader = _TorchStubLoader()

        def find_spec(
            self,
            fullname: str,
            path: object,
            target: object = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if fullname == "torch" or fullname.startswith("torch."):
                return importlib.machinery.ModuleSpec(
                    fullname,
                    self._loader,
                    origin="stub",
                    is_package=True,
                )
            return None

    sys.meta_path.insert(0, _TorchStubFinder())

    import torch  # noqa: PLC0415, F401  — populates sys.modules["torch"]
    """Register CUDA/cuDNN directories with the Windows DLL loader.

    On Windows, ``torch`` (imported transitively by ``modelscope`` which is a
    dependency of ``paddlex``) calls ``_load_dll_libraries()`` at import time.
    This requires cuDNN DLLs to be discoverable.  Calling
    ``os.add_dll_directory()`` for the CUDA bin path before the import ensures
    the loader can resolve them.

    This is a no-op on non-Windows platforms.
    """
    if os.name != "nt":
        return

    import glob as _glob  # noqa: PLC0415

    candidates: list[str] = []

    # 1. Prefer the env vars set by the CUDA installer.
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        val = os.environ.get(var)
        if val:
            candidates.append(os.path.join(val, "bin"))

    # 2. Fall back to scanning common NVIDIA install locations.
    for pattern in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin",
        r"C:\Program Files\NVIDIA\CUDA\v*\bin",
    ):
        candidates.extend(_glob.glob(pattern))

    registered: list[str] = []
    for d in candidates:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                registered.append(d)
            except OSError:
                pass

    if registered:
        logger.debug("Registered CUDA DLL dirs: %s", registered)
    else:
        logger.debug(
            "No CUDA DLL directories found. "
            "Set CUDA_PATH if torch fails to import."
        )


# ---------------------------------------------------------------------------
# Internal serialisation
# ---------------------------------------------------------------------------


def _result_to_raw(result: Any) -> dict[str, Any]:
    """Convert a ``PaddleOCRVLResult`` to a JSON-serializable dict.

    The ``parsing_res_list`` preserves the pipeline's reading order exactly.
    """
    blocks = []
    for idx, block in enumerate(result.get("parsing_res_list", [])):
        blocks.append(
            {
                "block_label": block.label,
                "block_content": block.content,
                "block_bbox": list(block.bbox),
                "block_id": idx,
            }
        )

    return {
        "page_index": result.get("page_index", 0),
        "width": result.get("width", 0),
        "height": result.get("height", 0),
        "parsing_res_list": blocks,
    }


# Verify protocol conformance at import time (zero runtime cost).
_: ParserBackend = PaddleVLBackend.__new__(PaddleVLBackend)
