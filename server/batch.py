"""Server-side Are.na batch roll job.

Pipelines download → infer → cook → publish so GPU/CPU work overlaps
kitchen eval + Are.na uploads. Repeat Arena blocks reuse the on-disk
infer cache instead of re-running the model.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from depth_core.infer_backend import (
    local_cook_slots,
    local_download_slots,
    local_infer_slots,
    resolve_backend,
)
from depth_core.pipeline import (
    DEFAULT_MODEL,
    finalize_from_arrays,
    normalize_model_key,
    run_rgb,
)
from depth_core.store import (
    compact_run,
    enforce_disk_budget,
    infer_cache_key,
    load_infer_cache,
    release_torch_memory,
)
from server.arena import (
    ArenaError,
    download_block_image,
    prepare_shuffle_pool,
    publish_image_to_channel,
)
from server.arena_export import build_arena_publish_fields
from server.cache import drop_heavy, pin_run, put_run, unpin_run
from server.graph_eval import evaluate_graph, graph_to_template, template_to_graph
from server.randomize import randomize_kitchen

SetJobFn = Callable[..., None]
GetJobFn = Callable[[str], Optional[dict[str, Any]]]

# Per-item fraction of overall progress by status.
_STAGE_PROGRESS = {
    "queued": 0.0,
    "downloading": 0.04,
    "ready": 0.12,
    "inferring": 0.2,
    "cooking": 0.55,
    "publishing": 0.78,
    "done": 1.0,
    "error": 1.0,
    "cancelled": 1.0,
}


def start_arena_roll_job(
    *,
    arena_url: str,
    count: int,
    model_key: str,
    max_size: int,
    infer_backend: Optional[str],
    wackiness: float,
    max_fidelity: bool,
    channel_id: str,
    title: str,
    graph: dict[str, Any],
    set_job: SetJobFn,
    get_job: GetJobFn,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    n = max(1, min(200, int(count)))
    key = normalize_model_key(model_key or DEFAULT_MODEL)
    backend = resolve_backend(infer_backend)
    fidelity = bool(max_fidelity)
    infer_sizes = (518, 756, 1008)
    effective_max_size = max(infer_sizes) if fidelity else int(max_size)
    items: list[dict[str, Any]] = [
        {
            "index": i + 1,
            "status": "queued",
            "title": None,
            "run_id": None,
            "arena_url": None,
            "error": None,
        }
        for i in range(n)
    ]
    set_job(
        job_id,
        status="running",
        progress=0.0,
        message="Queued…",
        kind="arena_roll",
        run_id=None,
        result=None,
        error=None,
        cancel=False,
        modal_billing={
            "cost_usd_so_far": 0.0,
            "gpu_seconds_so_far": 0.0,
            "rolls": 0,
        },
        phase="download",
        i=0,
        n=n,
        saved=0,
        downloaded=0,
        inferred=0,
        cached=0,
        cooked=0,
        published=0,
        items=list(items),
    )

    def worker() -> None:
        lock = threading.Lock()
        cost_so_far = 0.0
        gpu_so_far = 0.0
        saved = 0
        downloaded = 0
        inferred = 0
        cached = 0
        cooked = 0
        published = 0
        last_run_id: Optional[str] = None
        last_url: Optional[str] = None
        published_list: list[dict[str, Any]] = []
        cancel_flag = False
        temp_paths: list[Path] = []

        def cancelled() -> bool:
            nonlocal cancel_flag
            if cancel_flag:
                return True
            job = get_job(job_id) or {}
            if job.get("cancel"):
                cancel_flag = True
                return True
            return False

        def _progress() -> float:
            return sum(_STAGE_PROGRESS.get(it["status"], 0.0) for it in items) / max(
                n, 1
            )

        def _message() -> str:
            if cancel_flag:
                return f"Stopping… · saved {saved}/{n}"
            active = [
                it
                for it in items
                if it["status"] not in ("done", "error", "queued", "cancelled")
            ]
            head = (
                active[0]["status"]
                if active
                else ("done" if saved == n else "pipeline")
            )
            cache_bit = f" cache{cached}" if cached else ""
            return (
                f"Pipeline · ↓{downloaded}{cache_bit} GPU{inferred} "
                f"✎{cooked} ↑{published} / {n} · {head}"
            )

        def _phase() -> str:
            statuses = {it["status"] for it in items}
            if "publishing" in statuses:
                return "publish"
            if "cooking" in statuses:
                return "cook"
            if "inferring" in statuses:
                return "infer"
            if "downloading" in statuses or "ready" in statuses:
                return "download"
            if statuses <= {"done", "error", "cancelled"}:
                return "done"
            return "pipeline"

        def push(**extra: Any) -> None:
            with lock:
                payload = {
                    "status": "running",
                    "phase": _phase(),
                    "message": _message(),
                    "progress": round(_progress(), 4),
                    "i": published,
                    "n": n,
                    "saved": saved,
                    "downloaded": downloaded,
                    "inferred": inferred,
                    "cached": cached,
                    "cooked": cooked,
                    "published": published,
                    "items": [dict(it) for it in items],
                    "run_id": last_run_id,
                    "modal_billing": {
                        "cost_usd_so_far": round(cost_so_far, 6),
                        "gpu_seconds_so_far": round(gpu_so_far, 4),
                        "rolls": inferred if backend == "modal" else inferred,
                    },
                    "result": {
                        "saved": saved,
                        "last_block_url": last_url,
                        "published": list(published_list),
                        "run_id": last_run_id,
                    },
                }
                payload.update(extra)
                set_job(job_id, **payload)

        def set_item(i: int, **kw: Any) -> None:
            with lock:
                items[i].update(kw)

        def cook_and_publish(i: int, state: dict[str, Any], meta: dict[str, Any]) -> None:
            nonlocal saved, cooked, published, last_run_id, last_url
            if cancelled():
                set_item(i, status="cancelled")
                push()
                return

            set_item(i, status="cooking")
            push()
            try:
                run_id = put_run(state)
                pin_run(run_id)
                with lock:
                    last_run_id = run_id
                set_item(i, run_id=run_id)
                rolled = randomize_kitchen(
                    template_to_graph(template, run_id),
                    wackiness=wackiness,
                    max_fidelity=fidelity,
                )
                for node in rolled.get("nodes") or []:
                    if node.get("type") == "depth":
                        node.setdefault("data", {})["run_id"] = run_id
                evaluate_graph(run_id, rolled)
                with lock:
                    cooked += 1
                if cancelled():
                    set_item(i, status="cancelled")
                    push()
                    return

                fields = build_arena_publish_fields(
                    state,
                    graph=rolled,
                    user_title=title or None,
                    roll=(i + 1, n),
                )
                pub_title = fields["title"]
                set_item(i, status="publishing", title=pub_title)
                push()

                out = Path(state["out_dir"])
                blend = out / "blend_live.png"
                if not blend.is_file():
                    blend = out / f"{run_id}_blend.png"
                rgb = out / "rgb.jpg"
                if not rgb.is_file():
                    rgb = out / f"{run_id}_rgb.jpg"
                pub = publish_image_to_channel(
                    blend if blend.is_file() else rgb,
                    channel_id,
                    title=pub_title,
                    description=fields["description"],
                    original_source_url=fields.get("original_source_url"),
                    original_source_title=fields.get("original_source_title"),
                    metadata=fields.get("metadata"),
                )
                arena_block_url = (pub or {}).get("arena_url")
                with lock:
                    saved += 1
                    published += 1
                    last_url = arena_block_url or last_url
                    published_list.append(
                        {
                            "run_id": run_id,
                            "block": pub,
                            "title": pub_title,
                        }
                    )
                set_item(
                    i,
                    status="done",
                    title=pub_title,
                    arena_url=arena_block_url,
                    run_id=run_id,
                )
                push()
            except Exception as e:
                set_item(i, status="error", error=str(e))
                push()
                raise
            finally:
                out_dir = state.get("out_dir")
                if out_dir:
                    rid = Path(out_dir).name
                    compact_run(Path(out_dir))
                    drop_heavy(rid)
                    unpin_run(rid)

        try:
            template = graph_to_template(graph)
            push(
                phase="download",
                message=f"Picking {n} unique Are.na stills…",
            )

            pool = prepare_shuffle_pool(arena_url, n)
            channel_meta = dict(pool.meta)
            nested = channel_meta.get("nested_channels") or 0
            push(
                message=(
                    f"Rolling {n} Are.na stills"
                    + (f" · {nested} nested channels" if nested else "")
                    + "…"
                )
            )

            dl_workers = min(local_download_slots(), max(1, n))
            cook_workers = min(local_cook_slots(), max(1, n))
            infer_workers = (
                min(16, max(1, n))
                if backend == "modal"
                else min(local_infer_slots(), max(1, n))
            )
            pool_lock = threading.Lock()
            skipped = 0
            cook_errors: list[BaseException] = []

            def _next_block() -> Optional[dict[str, Any]]:
                with pool_lock:
                    return pool.take()

            def _on_cook_done(fut) -> None:
                exc = fut.exception()
                if exc is not None:
                    cook_errors.append(exc)

            def _infer_item(
                i: int,
                path: Optional[Path],
                resolved: dict[str, Any],
                cached_arrays: Optional[tuple[Any, Any]],
                ck: str,
            ) -> Optional[tuple[int, dict[str, Any], dict[str, Any]]]:
                nonlocal inferred, cached, cost_so_far, gpu_so_far
                if cancelled():
                    set_item(i, status="cancelled")
                    push()
                    return None
                block = resolved.get("block") or {}
                bid = block.get("id")
                meta = resolved
                set_item(i, status="inferring")
                push()
                try:
                    if cached_arrays is not None:
                        color, depth = cached_arrays
                        state = finalize_from_arrays(
                            color,
                            depth,
                            None,
                            key,
                            int(effective_max_size),
                            source_name=Path(f"arena_{bid or i}.jpg"),
                            infer_backend="cache",
                            infer_device="cache",
                            cache_key=ck,
                            arena_block_id=bid,
                        )
                        with lock:
                            cached += 1
                            inferred += 1
                    elif backend == "modal":
                        from depth_core.infer_modal import forward_rgb

                        assert path is not None
                        res = forward_rgb(path, key, int(effective_max_size))
                        billing = res.modal_billing or {}
                        with lock:
                            cost_so_far += float(billing.get("cost_usd") or 0)
                            gpu_so_far += float(billing.get("gpu_seconds") or 0)
                            inferred += 1
                        state = finalize_from_arrays(
                            res.color,
                            res.depth,
                            res.prompt_np,
                            res.model_key,
                            int(effective_max_size),
                            source_name=path,
                            infer_backend="modal",
                            infer_device=res.device,
                            modal_billing=billing,
                            cache_key=ck,
                            arena_block_id=bid,
                        )
                    else:
                        assert path is not None
                        state = run_rgb(
                            str(path),
                            key,
                            int(effective_max_size),
                            infer_backend="local",
                            arena_block_id=bid,
                            cache_key=ck,
                        )
                        with lock:
                            inferred += 1
                        release_torch_memory()
                finally:
                    if path is not None:
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            pass

                state["arena"] = {
                    "kind": meta.get("kind"),
                    "mode": meta.get("mode"),
                    "block": meta.get("block"),
                    "channel": meta.get("channel"),
                    "input": meta.get("input"),
                }
                push()
                return i, state, meta

            def _download_slot(i: int) -> tuple:
                nonlocal skipped, downloaded
                if cancelled():
                    raise ArenaError("Cancelled")
                set_item(i, status="downloading")
                push()
                while True:
                    block = _next_block()
                    if block is None:
                        raise ArenaError(
                            f"Could only fill {downloaded} unique still"
                            f"{'' if downloaded == 1 else 's'} "
                            f"(need {n}; skipped {skipped} broken URL"
                            f"{'' if skipped == 1 else 's'}; "
                            f"sampled {pool.sampled_count} from channel + nested)"
                        )
                    ck = infer_cache_key(
                        key,
                        int(effective_max_size),
                        arena_block_id=block.get("id"),
                    )
                    hit = load_infer_cache(ck)
                    resolved = {
                        "kind": "channel",
                        "input": arena_url.strip(),
                        "block": block,
                        "channel": channel_meta,
                        "imageable_count": pool.meta.get("sampled_stills"),
                        "mode": "shuffle",
                        "cached": hit is not None,
                    }
                    title_hint = block.get("title") or f"Roll {i + 1}"
                    if hit is not None:
                        set_item(i, status="ready", title=str(title_hint))
                        with lock:
                            downloaded += 1
                        push()
                        return i, None, resolved, hit, ck
                    try:
                        path = download_block_image(block)
                    except ArenaError:
                        with lock:
                            skipped += 1
                        continue
                    with lock:
                        temp_paths.append(path)
                        downloaded += 1
                    set_item(i, status="ready", title=str(title_hint))
                    push()
                    return i, path, resolved, None, ck

            with ThreadPoolExecutor(max_workers=cook_workers) as cook_pool:
                cook_futs: list = []
                cook_futs_lock = threading.Lock()
                with ThreadPoolExecutor(max_workers=infer_workers) as infer_pool:
                    infer_futs = []

                    def _infer_wrap(pack):
                        i = pack[0]
                        try:
                            return _infer_item(*pack)
                        except Exception as e:
                            set_item(i, status="error", error=str(e))
                            push()
                            raise

                    def _on_infer_done(fut) -> None:
                        try:
                            pack = fut.result()
                        except Exception as e:
                            cook_errors.append(e)
                            return
                        if pack is None:
                            return
                        i, state, meta = pack
                        cf = cook_pool.submit(cook_and_publish, i, state, meta)
                        cf.add_done_callback(_on_cook_done)
                        with cook_futs_lock:
                            cook_futs.append(cf)

                    with ThreadPoolExecutor(max_workers=dl_workers) as dl_pool:
                        fut_to_i = {
                            dl_pool.submit(_download_slot, i): i for i in range(n)
                        }
                        for fut in as_completed(fut_to_i):
                            i = fut_to_i[fut]
                            if cancelled():
                                for f in fut_to_i:
                                    f.cancel()
                                break
                            try:
                                pack = fut.result()
                            except Exception as e:
                                if cancelled():
                                    break
                                set_item(i, status="error", error=str(e))
                                push()
                                raise
                            inf = infer_pool.submit(_infer_wrap, pack)
                            inf.add_done_callback(_on_infer_done)
                            infer_futs.append(inf)

                    for fut in as_completed(infer_futs):
                        if cancelled():
                            break
                        _ = fut.exception()

                for fut in as_completed(cook_futs):
                    if cancelled():
                        break
                    exc = fut.exception()
                    if exc is not None and not isinstance(exc, ArenaError):
                        pass

            if cancelled():
                set_job(
                    job_id,
                    status="cancelled",
                    message=f"Stopped · saved {saved}/{n}",
                    progress=1.0,
                    saved=saved,
                    cached=cached,
                    items=[dict(it) for it in items],
                    phase="done",
                    run_id=last_run_id,
                    result={
                        "saved": saved,
                        "last_block_url": last_url,
                        "published": list(published_list),
                        "run_id": last_run_id,
                    },
                )
                return

            if cook_errors and saved == 0:
                raise cook_errors[0]

            protect = {last_run_id} if last_run_id else set()
            try:
                enforce_disk_budget(protect)
            except Exception:
                pass

            set_job(
                job_id,
                status="done",
                progress=1.0,
                message=f"Done · saved {saved}/{n}"
                + (f" · {cached} cache hits" if cached else ""),
                phase="done",
                saved=saved,
                downloaded=downloaded,
                inferred=inferred,
                cached=cached,
                cooked=cooked,
                published=published,
                items=[dict(it) for it in items],
                run_id=last_run_id,
                result={
                    "saved": saved,
                    "count": n,
                    "last_block_url": last_url,
                    "published": list(published_list),
                    "run_id": last_run_id,
                    "cached": cached,
                    "modal_billing": {
                        "cost_usd_so_far": round(cost_so_far, 6),
                        "gpu_seconds_so_far": round(gpu_so_far, 4),
                        "rolls": inferred,
                    },
                },
                modal_billing={
                    "cost_usd_so_far": round(cost_so_far, 6),
                    "gpu_seconds_so_far": round(gpu_so_far, 4),
                    "rolls": inferred,
                },
            )
        except ArenaError as e:
            set_job(
                job_id,
                status="error",
                progress=1.0,
                message=str(e),
                error=str(e),
                saved=saved,
                cached=cached,
                items=[dict(it) for it in items],
                phase="done",
            )
        except Exception as e:
            set_job(
                job_id,
                status="error",
                progress=1.0,
                message=str(e),
                error=f"{e}\n{traceback.format_exc()}",
                saved=saved,
                cached=cached,
                items=[dict(it) for it in items],
                phase="done",
            )
        finally:
            for p in temp_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            protect = {last_run_id} if last_run_id else set()
            try:
                enforce_disk_budget(protect)
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return job_id
