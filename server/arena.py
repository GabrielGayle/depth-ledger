"""Are.na helpers — v2 resolve/download + v3 auth/upload/publish."""

from __future__ import annotations

import json
import mimetypes
import os
import random
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

ARENA_API = "https://api.are.na/v2"  # legacy alias
ARENA_API_V2 = "https://api.are.na/v2"
ARENA_API_V3 = "https://api.are.na/v3"
USER_AGENT = "promptda-depth-generator/0.1 (+https://github.com/)"

# Still-image-bearing classes (original / display URL present).
IMAGEABLE = {"Image", "Link", "Media", "Attachment", "Embed"}
# Exclude video embeds / media and animated gifs from shuffle + depth.
VIDEO_EMBED_TYPES = {"video", "rich"}
VIDEO_URL_HINTS = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv",
    "tiktok.com",
    ".mp4",
    ".webm",
    ".mov",
    ".m4v",
)
MIN_EDGE_PX = 1200
NESTED_CHANNEL_MAX_DEPTH = 2
# Lazy shuffle: pull random Are.na pages (~50 items) until the pool needs more.
SHUFFLE_CHUNK = 50
SHUFFLE_PAGE_PER = 50

# In-memory lazy shuffle pools (cleared on process restart).
# key: normalized channel slug → ArenaShufflePool
_SHUFFLE_POOLS: dict[str, "ArenaShufflePool"] = {}
# Legacy full-list cache used only by non-shuffle resolve fallbacks.
_CHANNEL_CACHE: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}

_BLOCK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?are\.na/(?:block/)?(\d+)/?(?:[?#].*)?$",
    re.I,
)
_CHANNEL_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?are\.na/([^/?#]+)/([^/?#]+)/?(?:[?#].*)?$",
    re.I,
)
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$", re.I)


class ArenaError(ValueError):
    """User-facing Are.na resolve/download error."""


def arena_token() -> str:
    return (os.environ.get("ARENA_ACCESS_TOKEN") or "").strip()


def require_token() -> str:
    token = arena_token()
    if not token:
        raise ArenaError(
            "ARENA_ACCESS_TOKEN not set — add it to .env to publish / list channels"
        )
    return token


def _auth_headers(*, json_body: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    token = arena_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _request_json(
    method: str,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    body: Optional[dict[str, Any]] = None,
    timeout: float = 60,
) -> Any:
    qs = f"?{urllib.parse.urlencode(params)}" if params else ""
    full = f"{url}{qs}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        full,
        data=data,
        headers=_auth_headers(json_body=body is not None),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        raise ArenaError(f"Are.na API {e.code}: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise ArenaError(f"Are.na unreachable: {e.reason}") from e


def _get_json(path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """v2 GET helper (blocks / channels)."""
    return _request_json("GET", f"{ARENA_API_V2}{path}", params=params)


def _get_json_v3(path: str, params: Optional[dict[str, Any]] = None) -> Any:
    return _request_json("GET", f"{ARENA_API_V3}{path}", params=params)


def _post_json_v3(path: str, body: dict[str, Any]) -> Any:
    return _request_json("POST", f"{ARENA_API_V3}{path}", body=body)


def auth_status() -> dict[str, Any]:
    token = arena_token()
    if not token:
        return {"authenticated": False, "user": None}
    try:
        me = fetch_me()
        return {"authenticated": True, "user": me}
    except ArenaError as e:
        return {"authenticated": False, "user": None, "error": str(e)}


def fetch_me() -> dict[str, Any]:
    require_token()
    me = _get_json_v3("/me")
    return {
        "id": me.get("id"),
        "slug": me.get("slug"),
        "name": me.get("name"),
        "email": me.get("email"),
        "counts": me.get("counts") or {},
    }


def list_recent_channels(*, per: int = 24) -> list[dict[str, Any]]:
    """Recent channels owned/created by the authenticated user."""
    require_token()
    me = fetch_me()
    user_id = me.get("id")
    slug = me.get("slug")
    if not user_id:
        raise ArenaError("Could not resolve Are.na user id")
    per = max(1, min(100, int(per)))
    # Search ranks by updated_at more reliably than /contents for channels.
    data = _get_json_v3(
        "/search",
        {
            "q": "",
            "type": "Channel",
            "user_id": int(user_id),
            "per": per,
            "page": 1,
        },
    )
    items = data.get("data") if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for c in items or []:
        if not isinstance(c, dict):
            continue
        if (c.get("type") or "") != "Channel":
            continue
        owner = c.get("owner") or {}
        owner_slug = owner.get("slug") or slug or "channel"
        ch_slug = c.get("slug") or ""
        out.append(
            {
                "id": c.get("id"),
                "slug": ch_slug,
                "title": c.get("title") or ch_slug or str(c.get("id")),
                "visibility": c.get("visibility"),
                "updated_at": c.get("updated_at"),
                "arena_url": (
                    f"https://www.are.na/{owner_slug}/{ch_slug}"
                    if ch_slug
                    else f"https://www.are.na/channel/{c.get('id')}"
                ),
                "can_add": ((c.get("can") or {}) or {}).get("add_to", True),
            }
        )
    return out


def _guess_content_type(path: Path) -> str:
    ctype, _ = mimetypes.guess_type(str(path))
    if ctype:
        return ctype
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")


def upload_file_to_arena_temp(path: Path) -> str:
    """Presign + PUT local file; return public S3 URL for block creation."""
    require_token()
    path = Path(path)
    if not path.is_file():
        raise ArenaError(f"File not found: {path}")
    ctype = _guess_content_type(path)
    filename = path.name or "upload.png"
    pre = _post_json_v3(
        "/uploads/presign",
        {"files": [{"filename": filename, "content_type": ctype}]},
    )
    files = pre.get("files") if isinstance(pre, dict) else None
    if not files:
        raise ArenaError("Are.na presign returned no files")
    info = files[0]
    upload_url = info.get("upload_url")
    key = info.get("key")
    put_ctype = info.get("content_type") or ctype
    if not upload_url or not key:
        raise ArenaError("Are.na presign missing upload_url/key")

    data = path.read_bytes()
    req = urllib.request.Request(
        upload_url,
        data=data,
        headers={"Content-Type": put_ctype},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status not in (200, 201, 204):
                raise ArenaError(f"S3 upload failed ({resp.status})")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        raise ArenaError(f"S3 upload failed ({e.code}): {detail}") from e

    return f"https://s3.amazonaws.com/arena_images-temp/{key}"


def publish_image_to_channel(
    path: Path,
    channel_id: str | int,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    original_source_url: Optional[str] = None,
    original_source_title: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Upload local image and create a block connected to a channel."""
    require_token()
    channel_id = str(channel_id).strip()
    if not channel_id:
        raise ArenaError("Pick an Are.na channel")
    s3_url = upload_file_to_arena_temp(Path(path))
    ch_ref: Any = int(channel_id) if channel_id.isdigit() else channel_id
    payload: dict[str, Any] = {
        "value": s3_url,
        "channel_ids": [ch_ref],
    }
    if title:
        payload["title"] = title
    if description:
        payload["description"] = description
    if original_source_url:
        payload["original_source_url"] = original_source_url
    if original_source_title:
        payload["original_source_title"] = original_source_title
    if metadata:
        payload["metadata"] = metadata
    payload["alt_text"] = title or Path(path).name

    block = _post_json_v3("/blocks", payload)
    bid = block.get("id")
    return {
        "id": bid,
        "type": block.get("type") or block.get("base_type"),
        "state": block.get("state"),
        "title": block.get("title") or title or "",
        "arena_url": f"https://www.are.na/block/{bid}" if bid else None,
        "channel_id": channel_id,
        "original_source_url": original_source_url,
    }


def source_attribution_from_block(
    block: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build Are.na createBlock attribution fields from a source block summary."""
    if not isinstance(block, dict):
        return {}
    arena_url = (block.get("arena_url") or "").strip()
    if not arena_url and block.get("id") is not None:
        arena_url = f"https://www.are.na/block/{block['id']}"
    if not arena_url:
        return {}
    title = (block.get("title") or "").strip() or None
    out: dict[str, Any] = {
        "original_source_url": arena_url,
        "metadata": {"arena_source_block_url": arena_url},
    }
    if title:
        out["original_source_title"] = title
        out["metadata"]["arena_source_block_title"] = title
    if block.get("id") is not None:
        out["metadata"]["arena_source_block_id"] = block["id"]
    return out


def parse_arena_ref(raw: str) -> dict[str, str]:
    """Parse a paste of Are.na URL / block id / channel slug.

    Returns {"kind": "block"|"channel", "id": "..."}.
    """
    text = (raw or "").strip()
    if not text:
        raise ArenaError("Paste an are.na block or channel link")

    # Strip query/hash early for slug-only paths
    text = text.split("#")[0].strip()

    m = _BLOCK_RE.match(text)
    if m:
        return {"kind": "block", "id": m.group(1)}

    # Channel URL: are.na/user/slug  (ignore trailing /embed etc.)
    m = _CHANNEL_URL_RE.match(text.rstrip("/"))
    if m:
        user, slug = m.group(1), m.group(2)
        if user.lower() == "block":
            return {"kind": "block", "id": slug}
        # Drop known suffixes
        if slug.lower() in {"embed", "followers", "connections"}:
            slug = user  # unlikely; treat as channel slug alone
        return {"kind": "channel", "id": slug}

    if text.isdigit():
        return {"kind": "block", "id": text}

    # Bare channel slug
    slug = text.strip("/").split("/")[-1]
    if _SLUG_RE.match(slug):
        return {"kind": "channel", "id": slug}

    raise ArenaError("Unrecognized Are.na link — use a block or channel URL")


def _image_urls(block: dict[str, Any]) -> dict[str, Optional[str]]:
    image = block.get("image") or {}
    if not isinstance(image, dict):
        return {"original": None, "display": None, "thumb": None}

    def _url(key: str) -> Optional[str]:
        part = image.get(key)
        if isinstance(part, dict):
            u = part.get("url") or part.get("src")
            return str(u) if u else None
        return None

    # Some attachment / media shapes put file at top level
    file_url = None
    file_obj = block.get("attachment") or block.get("file")
    if isinstance(file_obj, dict):
        file_url = file_obj.get("url") or file_obj.get("src")

    # v3 ImageBlock uses image.src for the original; v2 uses image.original.url
    v3_src = image.get("src")
    original = _url("original") or (str(v3_src) if v3_src else None) or (
        str(file_url) if file_url else None
    )
    display = _url("display") or _url("large") or _url("medium")
    thumb = _url("thumb") or _url("square") or _url("small")

    return {
        "original": original,
        "display": display,
        "thumb": thumb,
    }


def _block_class(block: dict[str, Any]) -> str:
    return str(
        block.get("class")
        or block.get("type")
        or block.get("base_class")
        or "Unknown"
    )


def _is_channel_item(block: dict[str, Any]) -> bool:
    return _block_class(block) == "Channel" or block.get("base_type") == "Channel"


def _looks_like_gif(block: dict[str, Any]) -> bool:
    urls = _image_urls(block)
    for u in (urls.get("original"), urls.get("display"), urls.get("thumb")):
        if u and ".gif" in str(u).lower().split("?", 1)[0]:
            return True
    image = block.get("image") or {}
    ctype = str(image.get("content_type") or "").lower()
    if "gif" in ctype:
        return True
    attach = block.get("attachment") or {}
    if isinstance(attach, dict):
        act = str(attach.get("content_type") or attach.get("file_content_type") or "").lower()
        if "gif" in act:
            return True
        name = str(attach.get("file_name") or attach.get("filename") or "").lower()
        if name.endswith(".gif"):
            return True
    title = str(block.get("title") or block.get("generated_title") or "").lower()
    if title.endswith(".gif"):
        return True
    return False


def _looks_like_video(block: dict[str, Any]) -> bool:
    klass = _block_class(block)
    embed = block.get("embed") or {}
    if isinstance(embed, dict):
        et = str(embed.get("type") or "").lower()
        if et in VIDEO_EMBED_TYPES or et == "video":
            return True
        html = str(embed.get("html") or "").lower()
        if "youtube.com/embed" in html or "player.vimeo.com" in html:
            return True

    source = block.get("source") or {}
    src_url = ""
    if isinstance(source, dict):
        src_url = str(source.get("url") or "").lower()
        provider = ((source.get("provider") or {}) or {}).get("name") or ""
        if str(provider).lower() in {"youtube", "vimeo", "twitch", "tiktok"}:
            return True
    if any(h in src_url for h in VIDEO_URL_HINTS):
        return True

    attach = block.get("attachment") or {}
    if isinstance(attach, dict):
        act = str(attach.get("content_type") or attach.get("file_content_type") or "").lower()
        if act.startswith("video/"):
            return True
        name = str(attach.get("file_name") or attach.get("filename") or "").lower()
        if name.endswith((".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv")):
            return True

    # Media / Embed without a still-image original is usually video.
    if klass in {"Media", "Embed"} and not (_image_urls(block).get("original")):
        return True
    return False


def block_has_image(block: dict[str, Any]) -> bool:
    urls = _image_urls(block)
    return bool(urls["original"] or urls["display"])


def block_is_still_image(block: dict[str, Any]) -> bool:
    """True for downloadable stills — excludes nested channels, gif, video."""
    if _is_channel_item(block):
        return False
    if not block_has_image(block):
        return False
    if _looks_like_gif(block) or _looks_like_video(block):
        return False
    return True


def summarize_block(block: dict[str, Any]) -> dict[str, Any]:
    urls = _image_urls(block)
    klass = _block_class(block)
    original = urls["original"]
    display = urls["display"]
    thumb = urls["thumb"]
    return {
        "id": int(block["id"]),
        "class": klass,
        "title": block.get("title") or block.get("generated_title") or "",
        "has_image": block_has_image(block),
        "thumb_url": thumb or display or original,
        "display_url": display,
        # Prefer CDN/display over external originals (Link sources often rot).
        "image_url": display or original or thumb,
        "original_url": original,
        "source_url": ((block.get("source") or {}) or {}).get("url"),
        "arena_url": f"https://www.are.na/block/{block['id']}",
    }


def fetch_block(block_id: str | int) -> dict[str, Any]:
    if arena_token():
        try:
            return _get_json_v3(f"/blocks/{int(block_id)}")
        except ArenaError:
            pass
    return _get_json(f"/blocks/{int(block_id)}")


def fetch_channel(slug: str) -> dict[str, Any]:
    if arena_token():
        try:
            return _get_json_v3(f"/channels/{urllib.parse.quote(slug)}")
        except ArenaError:
            pass
    return _get_json(f"/channels/{urllib.parse.quote(slug)}")


def _channel_ref(item: dict[str, Any]) -> Optional[str]:
    slug = item.get("slug")
    if slug:
        return str(slug)
    cid = item.get("id")
    return str(cid) if cid is not None else None


def _channel_content_length(channel: dict[str, Any]) -> int:
    counts = channel.get("counts") or {}
    return int(counts.get("contents") or channel.get("length") or 0)


def _fetch_contents_page(
    slug: str, *, page: int = 1, per: int = SHUFFLE_PAGE_PER
) -> list[dict[str, Any]]:
    """Fetch one page of channel contents (v3 when authed, else v2)."""
    per = max(1, min(100, int(per)))
    page = max(1, int(page))
    if arena_token():
        try:
            data = _get_json_v3(
                f"/channels/{urllib.parse.quote(slug)}/contents",
                {"page": page, "per": per},
            )
            contents = data.get("data") if isinstance(data, dict) else data
            return [c for c in (contents or []) if isinstance(c, dict)]
        except ArenaError:
            pass
    data = _get_json(
        f"/channels/{urllib.parse.quote(slug)}",
        {"page": page, "per": per},
    )
    contents = data.get("contents") or []
    return [c for c in contents if isinstance(c, dict)]


class ArenaShufflePool:
    """Lazy unique still sampler across a channel + nested channels.

    Does **not** crawl every block up front. Pulls random pages of
    ``SHUFFLE_PAGE_PER`` items, keeps ~``SHUFFLE_CHUNK`` ready stills, and
    refills when ``take()`` needs more. Nested channel refs discovered on
    sampled pages (plus a light seed pass) are added as extra sources.
    """

    def __init__(self, root_slug: str, *, input_raw: str = ""):
        import threading

        self.root_slug = str(root_slug).strip()
        self.input_raw = input_raw or self.root_slug
        self._lock = threading.Lock()
        self.root_channel = fetch_channel(self.root_slug)
        self.meta = _channel_meta_from(
            self.root_channel,
            self.root_slug,
            _channel_content_length(self.root_channel),
        )
        # slug → {length, depth, tried_pages: set[int], empty_streak: int}
        self._sources: dict[str, dict[str, Any]] = {}
        self._seen_ids: set[int] = set()
        self._ready: list[dict[str, Any]] = []
        self._cursor = 0
        self._exhausted = False
        self._add_source(self.root_slug, self.root_channel, depth=0)
        self._seed_nested_channels()
        self.meta["nested_channels"] = max(0, len(self._sources) - 1)
        self.meta["imageable_count"] = None  # unknown until fully exhausted

    @classmethod
    def from_ref(cls, raw: str) -> "ArenaShufflePool":
        ref = parse_arena_ref(raw)
        if ref["kind"] == "block":
            raise ArenaError(
                "Batch roll needs an Are.na channel URL (not a single block)"
            )
        slug = str(ref["id"]).strip()
        # Fresh pool each time so within-roll uniqueness starts clean.
        pool = cls(slug, input_raw=raw.strip())
        _SHUFFLE_POOLS[slug.lower()] = pool
        return pool

    def _add_source(
        self, slug: str, channel: Optional[dict[str, Any]], *, depth: int
    ) -> None:
        key = str(slug).strip().lower()
        if not key or key in self._sources:
            return
        if depth > NESTED_CHANNEL_MAX_DEPTH:
            return
        ch = channel
        # Nested stubs in contents often lack counts — fetch meta when needed.
        if ch is None or _channel_content_length(ch) <= 0:
            try:
                ch = fetch_channel(slug)
            except ArenaError:
                if ch is None:
                    return
        length = _channel_content_length(ch or {})
        self._sources[key] = {
            "slug": (ch or {}).get("slug") or slug,
            "length": max(0, length),
            "depth": depth,
            "tried_pages": set(),
            "empty_streak": 0,
        }

    def _seed_nested_channels(self) -> None:
        """Light pass: a few random root pages to discover nested channel refs."""
        src = self._sources.get(self.root_slug.lower())
        if not src:
            return
        length = int(src["length"] or 0)
        pages = (
            max(1, (length + SHUFFLE_PAGE_PER - 1) // SHUFFLE_PAGE_PER)
            if length
            else 3
        )
        sample_n = min(3, pages)
        for page in random.sample(range(1, pages + 1), sample_n):
            try:
                items = _fetch_contents_page(
                    src["slug"], page=page, per=SHUFFLE_PAGE_PER
                )
            except ArenaError:
                continue
            self._ingest_page_items(items, source_depth=0, collect_stills=True)
        if self._ready:
            random.shuffle(self._ready)

    def _ingest_page_items(
        self,
        items: list[dict[str, Any]],
        *,
        source_depth: int,
        collect_stills: bool,
    ) -> int:
        gained = 0
        for item in items:
            if _is_channel_item(item):
                ref = _channel_ref(item)
                if ref:
                    self._add_source(ref, item, depth=source_depth + 1)
                continue
            if not collect_stills:
                continue
            bid = item.get("id")
            if bid is None:
                continue
            bid_i = int(bid)
            if bid_i in self._seen_ids:
                continue
            if not block_is_still_image(item):
                continue
            self._seen_ids.add(bid_i)
            self._ready.append(summarize_block(item))
            gained += 1
        return gained

    def _pick_source(self) -> Optional[dict[str, Any]]:
        open_sources = [
            s
            for s in self._sources.values()
            if int(s.get("empty_streak") or 0) < 4
        ]
        if not open_sources:
            return None
        weights = [max(1, int(s.get("length") or 1)) for s in open_sources]
        return random.choices(open_sources, weights=weights, k=1)[0]

    def _fetch_random_page(self, src: dict[str, Any]) -> list[dict[str, Any]]:
        length = int(src.get("length") or 0)
        pages = (
            max(1, (length + SHUFFLE_PAGE_PER - 1) // SHUFFLE_PAGE_PER)
            if length
            else 5
        )
        tried: set[int] = src["tried_pages"]
        choices = [p for p in range(1, pages + 1) if p not in tried]
        if not choices:
            if len(tried) >= pages:
                tried.clear()
            choices = list(range(1, pages + 1))
        page = random.choice(choices)
        tried.add(page)
        return _fetch_contents_page(src["slug"], page=page, per=SHUFFLE_PAGE_PER)

    def _refill_chunk(self) -> int:
        """Pull random pages until ~SHUFFLE_CHUNK new stills or sources look dry."""
        if self._exhausted:
            return 0
        start = len(self._ready)
        target = start + SHUFFLE_CHUNK
        gained_total = 0
        dry = 0
        max_attempts = max(24, SHUFFLE_CHUNK)
        for _ in range(max_attempts):
            if len(self._ready) >= target:
                break
            src = self._pick_source()
            if src is None:
                break
            try:
                items = self._fetch_random_page(src)
            except ArenaError:
                src["empty_streak"] = int(src.get("empty_streak") or 0) + 1
                dry += 1
                continue
            gained = self._ingest_page_items(
                items,
                source_depth=int(src.get("depth") or 0),
                collect_stills=True,
            )
            if gained:
                src["empty_streak"] = 0
                gained_total += gained
                dry = 0
            else:
                src["empty_streak"] = int(src.get("empty_streak") or 0) + 1
                dry += 1
            if dry >= 12 and gained_total == 0:
                break
        if gained_total and start < len(self._ready):
            tail = self._ready[start:]
            random.shuffle(tail)
            self._ready[start:] = tail
        self.meta["nested_channels"] = max(0, len(self._sources) - 1)
        self.meta["sampled_stills"] = len(self._seen_ids)
        if gained_total == 0 and self._cursor >= len(self._ready):
            self._exhausted = True
            self.meta["imageable_count"] = len(self._seen_ids)
        return gained_total

    def prime(self) -> int:
        """Ensure at least one chunk is buffered; return ready count."""
        with self._lock:
            if self._cursor >= len(self._ready):
                self._refill_chunk()
            return max(0, len(self._ready) - self._cursor)

    def take(self) -> Optional[dict[str, Any]]:
        """Pop next unique still summary, refilling in ~50 chunks as needed."""
        with self._lock:
            while self._cursor >= len(self._ready):
                if self._refill_chunk() == 0:
                    return None
            block = self._ready[self._cursor]
            self._cursor += 1
            return block

    @property
    def sampled_count(self) -> int:
        return len(self._seen_ids)


def _iter_channel_contents(slug: str, *, max_pages: int = 20, per: int = 100):
    """Yield content items from a channel (v3 when authed, else v2)."""
    per = max(1, min(100, per))
    if arena_token():
        try:
            channel = fetch_channel(slug)
            length = _channel_content_length(channel)
            pages = max(
                1, min(max_pages, (length + per - 1) // per if length else 5)
            )
            for page in range(1, pages + 1):
                data = _get_json_v3(
                    f"/channels/{urllib.parse.quote(slug)}/contents",
                    {"page": page, "per": per},
                )
                contents = data.get("data") if isinstance(data, dict) else data
                contents = contents or []
                if not contents:
                    break
                for item in contents:
                    if isinstance(item, dict):
                        yield item
                meta = data.get("meta") if isinstance(data, dict) else {}
                if meta and not meta.get("has_more_pages"):
                    break
                if len(contents) < per:
                    break
            return
        except ArenaError:
            pass

    channel = _get_json(f"/channels/{urllib.parse.quote(slug)}")
    length = int(channel.get("length") or 0)
    pages = max(1, min(max_pages, (length + per - 1) // per if length else 1))
    for page in range(1, pages + 1):
        data = _get_json(
            f"/channels/{urllib.parse.quote(slug)}",
            {"page": page, "per": per},
        )
        contents = data.get("contents") or []
        if not contents:
            break
        for item in contents:
            if isinstance(item, dict):
                yield item
        if len(contents) < per:
            break


def _channel_meta_from(channel: dict[str, Any], slug: str, block_count: int) -> dict[str, Any]:
    counts = channel.get("counts") or {}
    length = int(
        counts.get("contents")
        or channel.get("length")
        or block_count
        or 0
    )
    owner = (channel.get("owner") or channel.get("user") or {}) or {}
    user_slug = owner.get("slug") or owner.get("username") or "channel"
    ch_slug = channel.get("slug") or slug
    return {
        "id": channel.get("id"),
        "slug": ch_slug,
        "title": channel.get("title") or slug,
        "length": length,
        "user": user_slug,
        "arena_url": f"https://www.are.na/{user_slug}/{ch_slug}",
    }


def fetch_channel_image_blocks(
    slug: str,
    *,
    max_pages: int = 20,
    per: int = 100,
    recurse_nested: bool = True,
    max_depth: int = NESTED_CHANNEL_MAX_DEPTH,
    _visited: Optional[set[str]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (channel_meta, still-image blocks).

    When recurse_nested=True, also walks channels that live inside the channel
    (and one level deeper by default), collecting stills from nested channels.
    Skips gif and video Media/Embed blocks.
    """
    visited = _visited if _visited is not None else set()
    key = str(slug).strip().lower()
    if key in visited:
        return {"slug": slug, "title": slug, "length": 0, "user": "channel"}, []
    visited.add(key)

    channel = fetch_channel(slug)
    blocks: list[dict[str, Any]] = []
    seen: set[int] = set()
    nested_refs: list[str] = []

    for item in _iter_channel_contents(slug, max_pages=max_pages, per=per):
        if _is_channel_item(item):
            ref = _channel_ref(item)
            if ref and recurse_nested and max_depth > 0:
                nested_refs.append(ref)
            continue
        bid = item.get("id")
        if bid is None or int(bid) in seen:
            continue
        if not block_is_still_image(item):
            continue
        seen.add(int(bid))
        blocks.append(item)

    if recurse_nested and max_depth > 0:
        for ref in nested_refs:
            try:
                _, nested = fetch_channel_image_blocks(
                    ref,
                    max_pages=max_pages,
                    per=per,
                    recurse_nested=True,
                    max_depth=max_depth - 1,
                    _visited=visited,
                )
            except ArenaError:
                continue
            for item in nested:
                bid = item.get("id")
                if bid is None or int(bid) in seen:
                    continue
                seen.add(int(bid))
                blocks.append(item)

    meta = _channel_meta_from(channel, slug, len(blocks))
    meta["nested_channels"] = len(nested_refs)
    return meta, blocks


def resolve_ref(
    raw: str,
    *,
    mode: str = "auto",
) -> dict[str, Any]:
    """Resolve paste → metadata + selected block (if any).

    mode:
      - auto: block link → that block; channel → first still (incl. nested)
      - block: force block
      - shuffle: channel → random still from channel + nested channels
        (lazy ~50-at-a-time sampling; does not crawl every block)
    """
    ref = parse_arena_ref(raw)
    kind = ref["kind"]

    if kind == "block":
        block = fetch_block(ref["id"])
        summary = summarize_block(block)
        if not block_is_still_image(block):
            reason = "gif" if _looks_like_gif(block) else (
                "video" if _looks_like_video(block) else "no still image"
            )
            raise ArenaError(
                f"Block {summary['id']} is {summary['class']} ({reason}) — "
                "need a still image / link plate"
            )
        return {
            "kind": "block",
            "input": raw.strip(),
            "block": summary,
            "channel": None,
            "imageable_count": 1,
            "mode": "block",
            "cached": False,
        }

    slug = str(ref["id"]).strip()
    pick_mode = mode if mode in {"shuffle", "block", "auto"} else "auto"

    if pick_mode == "shuffle":
        pool = ArenaShufflePool.from_ref(raw)
        if pool.prime() <= 0:
            raise ArenaError(
                f"Channel “{pool.meta.get('title') or slug}” has no still images "
                "(gif / video excluded; checked nested channels too)"
            )
        chosen = pool.take()
        if chosen is None:
            raise ArenaError(
                f"Channel “{pool.meta.get('title') or slug}” has no still images "
                "(gif / video excluded; checked nested channels too)"
            )
        channel = {
            **pool.meta,
            "imageable_count": pool.meta.get("sampled_stills") or pool.sampled_count,
        }
        return {
            "kind": "channel",
            "input": raw.strip(),
            "block": chosen,
            "channel": channel,
            "imageable_count": channel.get("imageable_count"),
            "mode": "shuffle",
            "cached": False,
        }

    # auto / channel: first still (still uses fuller crawl for “first image” UX)
    channel, blocks = fetch_channel_image_blocks(slug, recurse_nested=True)
    if not blocks:
        raise ArenaError(
            f"Channel “{channel['title']}” has no still images "
            "(gif / video excluded; checked nested channels too)"
        )
    summaries = [summarize_block(b) for b in blocks]
    _CHANNEL_CACHE[slug.lower()] = (channel, summaries)
    chosen = summaries[0]
    return {
        "kind": "channel",
        "input": raw.strip(),
        "block": chosen,
        "channel": {
            **channel,
            "imageable_count": len(summaries),
        },
        "imageable_count": len(summaries),
        "mode": "channel",
        "cached": False,
    }


def resolve_shuffle_batch(raw: str, count: int) -> list[dict[str, Any]]:
    """Sample `count` unique stills lazily (refills in ~50 chunks)."""
    n = max(1, int(count))
    pool = prepare_shuffle_pool(raw, n)
    out: list[dict[str, Any]] = []
    for _ in range(n):
        block = pool.take()
        if block is None:
            title = pool.meta.get("title") or pool.root_slug
            raise ArenaError(
                f"Channel “{title}” only yielded {len(out)} unique still"
                f"{'' if len(out) == 1 else 's'} — need {n}"
            )
        out.append(
            {
                "kind": "channel",
                "input": raw.strip(),
                "block": block,
                "channel": pool.meta,
                "imageable_count": pool.meta.get("sampled_stills"),
                "mode": "shuffle",
                "cached": False,
            }
        )
    return out


def prepare_shuffle_pool(raw: str, count: int) -> ArenaShufflePool:
    """Build a lazy unique still pool for batch rolls (channel + nested).

    Pulls random ~50-item pages from the root channel and discovered
    subchannels, refilling when ``take()`` needs more. Does not require
    knowing the full still count up front.
    """
    _ = max(1, int(count))  # reserved for future prefill sizing
    pool = ArenaShufflePool.from_ref(raw)
    if pool.prime() <= 0:
        title = pool.meta.get("title") or pool.root_slug
        raise ArenaError(
            f"Channel “{title}” has no still images "
            "(gif / video excluded; checked nested channels too)"
        )
    return pool


def _ensure_min_edge(path: Path, min_edge: int = MIN_EDGE_PX) -> Path:
    """Upscale so the longest edge is at least min_edge (no downscale)."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ArenaError("Pillow required to upscale Are.na images") from e

    with Image.open(path) as im:
        im.load()
        w, h = im.size
        long_edge = max(w, h)
        if long_edge >= min_edge:
            return path
        scale = min_edge / float(long_edge)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        out = im.convert("RGB") if im.mode not in ("RGB", "L") else im.copy()
        if out.mode == "L":
            out = out.convert("RGB")
        out = out.resize((nw, nh), resample)

    # Always write JPEG after upscale for a stable Depth Pro input.
    dest = path if path.suffix.lower() in {".jpg", ".jpeg"} else path.with_suffix(".jpg")
    out.save(dest, format="JPEG", quality=92, optimize=True)
    if dest != path and path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    return dest


def _normalize_image_url(url: Optional[str]) -> Optional[str]:
    """Make a downloadable absolute http(s) URL, or None if unusable."""
    if not url:
        return None
    u = str(url).strip()
    if not u or u.lower() in {"none", "null", "undefined"}:
        return None
    if u.startswith("//"):
        u = "https:" + u
    elif u.startswith("/"):
        # Relative Are.na asset path — rare, but recoverable.
        u = "https://www.are.na" + u
    parsed = urllib.parse.urlparse(u)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").strip()
    if not host:
        return None
    return u


def _arena_cdn_score(url: str) -> int:
    """Higher = prefer for download (Are.na CDN over brittle external hosts)."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host.endswith("are.na") or "are.na" in host:
        return 3
    if "cloudfront" in host or host.endswith("amazonaws.com"):
        return 2
    return 1


def _block_image_candidates(block_summary: dict[str, Any]) -> list[str]:
    # Prefer full-res original when it's on a CDN; else Are.na display/thumb.
    raw = [
        block_summary.get("original_url"),
        block_summary.get("display_url"),
        block_summary.get("image_url"),
        block_summary.get("thumb_url"),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in raw:
        u = _normalize_image_url(candidate if isinstance(candidate, str) else None)
        if not u or u in seen:
            continue
        seen.add(u)
        ordered.append(u)
    return sorted(
        ordered,
        key=lambda u: (-_arena_cdn_score(u), ordered.index(u)),
    )


def download_image(url: str, dest: Optional[Path] = None) -> Path:
    """Download image bytes to a temp (or given) path; return path."""
    normalized = _normalize_image_url(url)
    if not normalized:
        raise ArenaError("No image URL to download")
    req = urllib.request.Request(
        normalized,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/*,*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    except urllib.error.HTTPError as e:
        raise ArenaError(f"Image download failed ({e.code})") from e
    except urllib.error.URLError as e:
        reason = e.reason
        raise ArenaError(f"Image download failed: {reason}") from e
    except OSError as e:
        raise ArenaError(f"Image download failed: {e}") from e

    if len(data) < 32:
        raise ArenaError("Downloaded file too small to be an image")

    if "gif" in ctype or str(normalized).lower().split("?", 1)[0].endswith(".gif"):
        raise ArenaError("GIF blocks are excluded — pick a still image")

    ext = ".jpg"
    if "png" in ctype:
        ext = ".png"
    elif "webp" in ctype:
        ext = ".webp"
    elif "jpeg" in ctype or "jpg" in ctype:
        ext = ".jpg"
    else:
        path_ext = Path(urllib.parse.urlparse(normalized).path).suffix.lower()
        if path_ext in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = path_ext if path_ext != ".jpeg" else ".jpg"

    if dest is None:
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        dest = Path(tmp.name)
        tmp.write(data)
        tmp.close()
    else:
        dest = Path(dest)
        dest.write_bytes(data)
    return _ensure_min_edge(dest, MIN_EDGE_PX)


def download_block_image(block_summary: dict[str, Any]) -> Path:
    candidates = _block_image_candidates(block_summary)
    if not candidates:
        raise ArenaError("Selected block has no image URL")
    last_err: Optional[Exception] = None
    for url in candidates:
        try:
            return download_image(url)
        except ArenaError as e:
            last_err = e
            continue
    bid = block_summary.get("id")
    detail = str(last_err) if last_err else "unknown error"
    raise ArenaError(
        f"Could not download block {bid or '?'} ({len(candidates)} URL"
        f"{'' if len(candidates) == 1 else 's'} tried): {detail}"
    )
