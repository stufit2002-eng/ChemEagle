"""
ChemEagle Pipeline Trace Logger
================================
Thread-safe, human-readable trace log for every model inference call
(RxnIM, MolNexTR, ChemIE, Graph2SMILES, ChemNER) and every LLM API call
(AzureOpenAI / OpenAI chat.completions) that occurs during a pipeline run.

Usage
-----
Each entry point activates tracing before the pipeline runs:

    from trace_logger import TRACER
    TRACER.initialize(entry_point="main", source_name="reaction1.jpg")
    # ... run pipeline ...
    TRACER.close()          # optional; flushes and closes the file

At every model / LLM call site the trace is recorded via:

    result = TRACER.model_call("RxnIM.predict_image_file",
                               "get_reaction_agent.py › get_reaction",
                               {"image_path": img, "molnextr": True},
                               lambda: model1.predict_image_file(img, ...))

    response = TRACER.llm_call("ChemEagle [planner] / gpt-5-mini",
                                "main.py › ChemEagle",
                                {"model": "gpt-5-mini", "messages": msgs},
                                lambda: client.chat.completions.create(...))

When TRACER is not initialized (not enabled), every call is a zero-overhead
pass-through — the lambda is called directly with no logging overhead.
"""

import io
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── Visual constants ────────────────────────────────────────────────────────
_W          = 76                        # total line width
_SEP_WIDE   = "═" * _W
_SEP_THIN   = "─" * _W
_SMILES_MAX = 80    # max SMILES chars displayed inline
_TEXT_MAX   = 300   # max plain-text string before truncation
_RESP_MAX   = 1200  # max LLM response content chars


# ════════════════════════════════════════════════════════════════════════════
#  Tracer
# ════════════════════════════════════════════════════════════════════════════

class ChemEagleTracer:
    """Thread-safe trace logger for the full ChemEagle inference pipeline."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._file: Optional[io.TextIOWrapper] = None
        self._counter = 0
        self._enabled = False
        self._log_path: Optional[str] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def initialize(
        self,
        log_path: Optional[str] = None,
        entry_point: str = "",
        source_name: str = "",
    ) -> str:
        """Open a new trace log file and enable tracing.

        Args:
            log_path:     Explicit output path.  If *None*, a timestamped
                          path inside ``./trace_logs/`` is generated.
            entry_point:  Short label for the entry script, e.g. ``"main"``
                          or ``"pdf"`` or ``"streamlit"``.
            source_name:  Human-readable source, e.g. image filename or PDF
                          name.  Used in the auto-generated filename.

        Returns:
            The resolved log-file path (useful for displaying to the user).
        """
        if log_path is None:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            ep  = (entry_point or "run").replace(" ", "_")
            src = Path(source_name).stem if source_name else "unknown"
            os.makedirs("./trace_logs", exist_ok=True)
            log_path = f"./trace_logs/trace_{ts}_{ep}_{src}.log"

        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
            self._file    = open(log_path, "w", encoding="utf-8")
            self._counter = 0
            self._enabled = True
            self._log_path = log_path

        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._raw_write(
            f"\n{_SEP_WIDE}\n"
            f"  ChemEagle Pipeline Trace Log\n"
            f"  Started    : {ts_human}\n"
            f"  Entry point: {entry_point or '(unknown)'}\n"
            f"  Source     : {source_name or '(unknown)'}\n"
            f"  Log file   : {log_path}\n"
            f"{_SEP_WIDE}\n"
        )
        return log_path

    def close(self) -> None:
        """Flush and close the trace log."""
        with self._lock:
            if self._file:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._file.write(
                    f"\n{_SEP_WIDE}\n"
                    f"  Trace complete · {ts}\n"
                    f"{_SEP_WIDE}\n"
                )
                self._file.close()
                self._file = None
            self._enabled = False

    @property
    def log_path(self) -> Optional[str]:
        return self._log_path

    # ── Public tracing API ───────────────────────────────────────────────────

    def model_call(
        self,
        label: str,
        caller: str,
        input_kwargs: Dict[str, Any],
        thunk: Callable[[], Any],
    ) -> Any:
        """Trace a model inference call.

        Args:
            label:        Model name + method, e.g. ``"RxnIM.predict_image_file"``.
            caller:       File › function chain for the call site.
            input_kwargs: Dict of arguments to log (paths, flags, etc.).
            thunk:        Zero-argument callable that performs the inference.
        """
        if not self._enabled:
            return thunk()

        call_id = self._next_id()
        ts      = _now_ms()
        thread  = threading.current_thread().name

        lines = [
            f"\n{_SEP_WIDE}",
            f"  CALL #{call_id:04d} · {ts} · Thread: {thread}",
            f"  KIND   : MODEL — {label}",
            f"  CALLER : {caller}",
            _SEP_THIN,
            "  INPUT:",
        ]
        for k, v in input_kwargs.items():
            lines.append(f"    {k:<14} = {_fmt_value(v)}")
        self._write_lines(lines)

        t0 = time.perf_counter()
        try:
            result  = thunk()
            elapsed = time.perf_counter() - t0
            out = [_SEP_THIN, "  OUTPUT:"]
            out += _fmt_model_output(label, result)
            out += [_SEP_THIN, f"  ELAPSED : {elapsed:.3f}s", _SEP_WIDE]
            self._write_lines(out)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self._write_lines([
                _SEP_THIN,
                f"  ✗ ERROR  : {type(exc).__name__}: {exc}",
                _SEP_THIN,
                f"  ELAPSED : {elapsed:.3f}s",
                _SEP_WIDE,
            ])
            raise

        return result

    def llm_call(
        self,
        label: str,
        caller: str,
        api_kwargs: Dict[str, Any],
        thunk: Callable[[], Any],
    ) -> Any:
        """Trace an LLM chat-completions call.

        Args:
            label:      Short label, e.g.
                        ``"ChemEagle [planner] / gpt-5-mini"``.
            caller:     File › function chain.
            api_kwargs: Dict with at minimum ``model`` and ``messages`` keys
                        (as passed to ``chat.completions.create``).  Used for
                        logging only — the actual call is made via *thunk*.
            thunk:      Zero-argument callable that performs the API call and
                        returns the response object.
        """
        if not self._enabled:
            return thunk()

        call_id = self._next_id()
        ts      = _now_ms()
        thread  = threading.current_thread().name

        lines = [
            f"\n{_SEP_WIDE}",
            f"  CALL #{call_id:04d} · {ts} · Thread: {thread}",
            f"  KIND   : LLM — {label}",
            f"  CALLER : {caller}",
            _SEP_THIN,
        ]
        lines += _fmt_llm_messages(api_kwargs.get("messages", []))

        tools = api_kwargs.get("tools")
        if tools:
            lines.append(_SEP_THIN)
            lines.append(f"  TOOLS ({len(tools)}):")
            for t in tools:
                fn  = t.get("function", {})
                nm  = fn.get("name", "?")
                dsc = fn.get("description", "")[:80]
                lines.append(f"    · {nm} — {dsc}")
        else:
            lines += [_SEP_THIN, "  TOOLS : None"]
        self._write_lines(lines)

        t0 = time.perf_counter()
        try:
            response = thunk()
            elapsed  = time.perf_counter() - t0
            out = [_SEP_THIN, "  RESPONSE:"]
            out += _fmt_llm_response(response)
            usage = getattr(response, "usage", None)
            if usage:
                p = getattr(usage, "prompt_tokens",     "?")
                c = getattr(usage, "completion_tokens", "?")
                tt= getattr(usage, "total_tokens",      "?")
                out.append(_SEP_THIN)
                out.append(f"  USAGE   : prompt={p}  completion={c}  total={tt}")
            out += [_SEP_THIN, f"  ELAPSED : {elapsed:.3f}s", _SEP_WIDE]
            self._write_lines(out)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self._write_lines([
                _SEP_THIN,
                f"  ✗ ERROR  : {type(exc).__name__}: {exc}",
                _SEP_THIN,
                f"  ELAPSED : {elapsed:.3f}s",
                _SEP_WIDE,
            ])
            raise

        return response

    def section(self, title: str, **meta) -> None:
        """Write a visual section header (e.g. start of a crop/image run)."""
        if not self._enabled:
            return
        ts     = _now_ms()
        thread = threading.current_thread().name
        w      = _W - 4
        meta_str = "  ".join(f"{k}={v}" for k, v in meta.items())
        lines = [
            f"\n{'┌' + '─' * (_W - 2) + '┐'}",
            f"│  SECTION : {title:<{w - 11}}│",
        ]
        if meta_str:
            lines.append(f"│  {meta_str:<{w}}│")
        lines.append(f"│  {ts}  Thread: {thread:<{w - len(ts) - 10}}│")
        lines.append(f"{'└' + '─' * (_W - 2) + '┘'}")
        self._write_lines(lines)

    def note(self, msg: str) -> None:
        """Write a free-form note line (info, warnings, quality checks …)."""
        if not self._enabled:
            return
        ts = _now_ms()
        self._raw_write(f"  ℹ  [{ts}] {msg}\n")

    # ── Internals ────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _write_lines(self, lines: List[str]) -> None:
        self._raw_write("\n".join(lines) + "\n")

    def _raw_write(self, text: str) -> None:
        with self._lock:
            if self._file:
                self._file.write(text)
                self._file.flush()


# ── Module-level singleton ───────────────────────────────────────────────────
TRACER = ChemEagleTracer()


# ════════════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ════════════════════════════════════════════════════════════════════════════

def _now_ms() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _fmt_value(v: Any, max_len: int = _TEXT_MAX) -> str:
    """One-line representation of a value, with base64 redaction."""
    if isinstance(v, str):
        if len(v) > 100 and _is_base64_blob(v):
            kb = len(v) * 3 // 4 // 1024
            return f"[base64-blob  {len(v)} chars ≈ {kb} kB]"
        s = v[:max_len]
        return repr(s) + ("…" if len(v) > max_len else "")
    if isinstance(v, (list, tuple)):
        return f"[{type(v).__name__}  {len(v)} item(s)]"
    if isinstance(v, dict):
        return f"{{dict  {len(v)} key(s): {list(v.keys())[:6]}}}"
    return repr(v)


def _is_base64_blob(s: str) -> bool:
    return bool(re.match(r'^[A-Za-z0-9+/=]{100,}$', s[:200]))


def _redact_base64(text: str) -> str:
    """Replace data-URI base64 blobs with a short summary."""
    return re.sub(
        r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{40,}',
        lambda m: f'[base64-image  ≈{len(m.group())//1024} kB]',
        text,
    )


# ── Model output formatters ──────────────────────────────────────────────────

def _fmt_model_output(label: str, result: Any) -> List[str]:
    """Dispatch to the right formatter based on the model label."""
    lbl = label.lower()
    if "rxnim" in lbl or "predict_image" in lbl or "moldetect" in lbl:
        return _fmt_rxnim_output(result)
    if "extract_molecule_corefs" in lbl or "chemie" in lbl:
        return _fmt_coref_output(result)
    if "graph2smiles" in lbl or "_convert_graph" in lbl:
        return _fmt_g2s_output(result)
    if "chemner" in lbl or "predict_strings" in lbl:
        return _fmt_ner_output(result)
    # Generic fallback
    s = str(result)
    return [f"    {s[:400]}" + ("…" if len(s) > 400 else "")]


def _fmt_rxnim_output(result: Any) -> List[str]:
    """Format RxnIM / predict_image_file output (list of reaction dicts)."""
    lines = []
    if not isinstance(result, list):
        lines.append(f"    {type(result).__name__}: {str(result)[:200]}")
        return lines
    lines.append(f"    [list]  {len(result)} reaction(s)")
    for i, rxn in enumerate(result):
        if not isinstance(rxn, dict):
            lines.append(f"    [{i}] {str(rxn)[:100]}")
            continue
        r = len(rxn.get("reactants",  []))
        c = len(rxn.get("conditions", []))
        p = len(rxn.get("products",   []))
        lines.append(f"    [{i}] reactants={r}  conditions={c}  products={p}")
        for sec in ("reactants", "products"):
            for j, item in enumerate(rxn.get(sec, [])[:3]):
                sm = item.get("smiles", "")
                if sm:
                    sm_disp = sm[:_SMILES_MAX] + ("…" if len(sm) > _SMILES_MAX else "")
                    lines.append(f"          {sec}[{j}].smiles = {sm_disp}")
        for j, cond in enumerate(rxn.get("conditions", [])[:3]):
            if "text" in cond and cond["text"]:
                lines.append(f"          conditions[{j}].text   = {cond['text']}")
            elif "smiles" in cond:
                lines.append(f"          conditions[{j}].smiles = {cond['smiles'][:60]}")
    return lines


def _fmt_coref_output(result: Any) -> List[str]:
    """Format extract_molecule_corefs_from_figures output."""
    lines = []
    if not isinstance(result, list):
        lines.append(f"    {type(result).__name__}: {str(result)[:200]}")
        return lines
    lines.append(f"    [list]  {len(result)} coref block(s)")
    for i, block in enumerate(result):
        bboxes = block.get("bboxes", []) if isinstance(block, dict) else []
        lines.append(f"    [{i}] {len(bboxes)} bbox(es)")
        for j, bbox in enumerate(bboxes[:5]):
            cat    = bbox.get("category", "?")
            smiles = bbox.get("smiles", "")
            text   = bbox.get("text", [])
            if smiles:
                sm_disp = smiles[:_SMILES_MAX] + ("…" if len(smiles) > _SMILES_MAX else "")
                lines.append(f"          [{j}] {cat:10s} smiles={sm_disp}")
            elif text:
                lines.append(f"          [{j}] {cat:10s} text={text[:3]}")
        if len(bboxes) > 5:
            lines.append(f"          … and {len(bboxes)-5} more")
    return lines


def _fmt_g2s_output(result: Any) -> List[str]:
    """Format _convert_graph_to_smiles output: (smiles, molfile, n_trials)."""
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        sm       = str(result[0])
        n_trials = result[2] if len(result) > 2 else "?"
        sm_disp  = sm[:_SMILES_MAX] + ("…" if len(sm) > _SMILES_MAX else "")
        return [f"    smiles   = {sm_disp}", f"    n_trials = {n_trials}"]
    return [f"    {result!r}"]


def _fmt_ner_output(result: Any) -> List[str]:
    """Format ChemNER predict_strings output."""
    if isinstance(result, list):
        lines = [f"    [list]  {len(result)} prediction(s)"]
        for i, pred in enumerate(result[:5]):
            lines.append(f"    [{i}] {str(pred)[:120]}")
        if len(result) > 5:
            lines.append(f"    … and {len(result)-5} more")
        return lines
    return [f"    {type(result).__name__}: {str(result)[:200]}"]


# ── LLM output formatters ────────────────────────────────────────────────────

def _fmt_llm_messages(messages: list) -> List[str]:
    """Format a messages list for the log, redacting base64 images."""
    if not messages:
        return ["  MESSAGES : (none)"]
    lines = [f"  MESSAGES ({len(messages)}):"]
    for i, msg in enumerate(messages):
        role    = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            display = _redact_base64(content)[:_TEXT_MAX]
            if len(content) > _TEXT_MAX:
                display += "…"
            lines.append(f"    [{i}] role={role}")
            lines.append(f"        {display!r}")
        elif isinstance(content, list):
            lines.append(f"    [{i}] role={role}  [{len(content)} part(s)]")
            for j, part in enumerate(content):
                ptype = part.get("type", "?")
                if ptype == "text":
                    txt     = part.get("text", "")
                    display = txt[:_TEXT_MAX] + ("…" if len(txt) > _TEXT_MAX else "")
                    lines.append(f"        part[{j}] text ({len(txt)} chars)")
                    # Show first 2 lines of text content
                    for line in display.split("\n")[:3]:
                        lines.append(f"                 {line}")
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        semi   = url.index(";") if ";" in url else 20
                        mime   = url[5:semi]
                        b64    = url.split(",", 1)[1] if "," in url else ""
                        size_kb = len(b64) * 3 // 4 // 1024
                        lines.append(f"        part[{j}] image ({mime}  ≈{size_kb} kB)")
                    else:
                        lines.append(f"        part[{j}] image_url: {url[:100]}")
                else:
                    lines.append(f"        part[{j}] {ptype}: {str(part)[:80]}")
        else:
            lines.append(f"    [{i}] role={role}  {str(content)[:100]}")
    return lines


def _fmt_llm_response(response: Any) -> List[str]:
    """Format an OpenAI-style response object."""
    lines = []
    if not response:
        lines.append("    (empty response)")
        return lines
    choices = getattr(response, "choices", [])
    if not choices:
        lines.append("    choices: []")
        return lines
    choice = choices[0]
    finish = getattr(choice, "finish_reason", "?")
    msg    = getattr(choice, "message", None)
    lines.append(f"    finish_reason : {finish}")
    if msg is None:
        lines.append("    message       : None")
        return lines

    # Tool calls
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        lines.append(f"    tool_calls ({len(tool_calls)}):")
        for tc in tool_calls:
            fn   = getattr(tc, "function", None)
            name = getattr(fn, "name",      "?") if fn else "?"
            args = getattr(fn, "arguments", "") if fn else ""
            disp = args[:250] + ("…" if len(args) > 250 else "")
            lines.append(f"      · {name}({disp})")
    else:
        content = getattr(msg, "content", "") or ""
        disp    = content[:_RESP_MAX] + ("…" if len(content) > _RESP_MAX else "")
        lines.append(f"    content ({len(content)} chars):")
        for line in disp.split("\n"):
            lines.append(f"      {line}")
    return lines
