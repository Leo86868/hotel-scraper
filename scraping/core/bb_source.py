"""bb-browser daemon HTTP API image extraction + download.

Uses daemon → Chrome Extension → chrome.debugger path (the ONLY path where clicks work).
DO NOT use subprocess calls to bb-browser CLI.
"""
import json, logging, os, re, time, uuid
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import requests as http_requests
from scraping.config import BB_DAEMON_URL, GOOGLE_IMAGES_MAX, OFFICIAL_SITE_MAX

logger = logging.getLogger(__name__)
_DL_TIMEOUT = 15
_DL_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36", "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_JUNK = ["favicon", "logo", "icon", "pixel", "spacer", "tracking", "1x1", "sprite", "svg", "badge", "avatar", "star", "rating", "/a/", "/a-/",
         "maps.googleapis.com", "maps.gstatic.com", "khms", "tile.openstreetmap", "maps/api/", "map-tile"]
_current_tab_id: Optional[int] = None

def cmd(action: str, timeout: int = 20, **kw) -> dict:
    """POST a command to the bb-browser daemon. Auto-attaches current tabId."""
    payload = {"id": str(uuid.uuid4()), "action": action, **kw}
    if _current_tab_id and "tabId" not in kw:
        payload["tabId"] = _current_tab_id
    try:
        return http_requests.post(f"{BB_DAEMON_URL}/command", json=payload, timeout=timeout).json()
    except http_requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to bb-browser daemon at {BB_DAEMON_URL}")
    except http_requests.exceptions.Timeout:
        return {"success": False, "error": "timeout"}

def _open(url: str):
    global _current_tab_id
    r = cmd("open", url=url)
    if r.get("success"): _current_tab_id = r["data"].get("tabId")

def _eval(expr: str, timeout: int = 25) -> str:
    r = cmd("eval", timeout=timeout, script=expr)
    if r.get("success"):
        v = r.get("data", {}).get("result", r.get("data", {}).get("value", ""))
        return str(v) if v is not None else ""
    return ""

def _close():
    global _current_tab_id
    cmd("close")
    _current_tab_id = None

def _snap() -> str:
    r = cmd("snapshot", interactive=True, timeout=30)
    if not r.get("success"): return ""
    sd = r.get("data", {}).get("snapshotData", "")
    return sd.get("snapshot", "") if isinstance(sd, dict) else str(sd)

def _check_connection():
    try:
        s = http_requests.get(f"{BB_DAEMON_URL}/status", timeout=5).json()
        if not s.get("running"): raise ConnectionError("bb-browser daemon not running")
        if not s.get("extensionConnected"): raise ConnectionError("Chrome extension not connected")
    except http_requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to bb-browser daemon at {BB_DAEMON_URL}")

def _is_junk(url: str) -> bool:
    return any(p in url.lower() for p in _JUNK)

def _extract_google_images(poi_name: str) -> List[dict]:
    encoded = (poi_name + " hotel").replace(" ", "+").replace("&", "%26")
    _open(f"https://www.google.com/search?q={encoded}&udm=2")
    time.sleep(5)
    # Google hides original URLs in <script> tags — extract all, filter in Python
    raw = _eval(
        "(function(){var t='';document.querySelectorAll('script').forEach(function(s){t+=s.textContent;});"
        "var m=t.match(/https:\\/\\/[^\\s'\"\\\\]{20,500}\\.(?:jpg|jpeg|png|webp)[^\\s'\"\\\\]*/gi)||[];"
        "var s=new Set(),u=[];for(var i=0;i<m.length;i++){var v=m[i];if(!s.has(v)){s.add(v);u.push(v);}}"
        "return JSON.stringify(u);})()", timeout=30)
    _skip_domains = ["google.", "gstatic.", "googleapis.", "schema.org", "youtube.", "blogger."]
    entries = []
    try:
        urls = json.loads(raw) if raw else []
        for url in urls:
            if len(entries) >= GOOGLE_IMAGES_MAX:
                break
            url = url.replace("\\u003d", "=").replace("\\u0026", "&")
            if any(d in url for d in _skip_domains):
                continue
            if not _is_junk(url):
                entries.append({"url": url, "source": "google_images"})
    except (json.JSONDecodeError, TypeError):
        logger.warning("Google Images: parse failed")
    _close()
    logger.info("Google Images: %d URLs for '%s'", len(entries), poi_name)
    return entries

def _boost_cdn(url: str) -> str:
    if "cache.marriott.com/is/image/" in url:
        return re.sub(r'wid=\d+', 'wid=2000', url)
    if "marriott-renditions" in url and "downsize=" in url:
        return re.sub(r'downsize=\d+px:\*', '', url).rstrip('&?')
    if "fourseasons.com/alt/img-opt/" in url:
        return re.sub(r'~\d+\.\d+\.', '~90.2000.', url)
    return url

_SERP_NOISE = {"About this result", "Search Labs", "Google apps", "Google Account",
                "More filters", "Tools", "Search", "Page 2", "Page 3", "Page 4",
                "Page 5", "Page 6", "Page 7", "Page 8", "Page 9", "Page 10",
                "How much", "Are meals", "What is the least", "Is ", "More info"}


def _clean_serp_snapshot(snap: str) -> str:
    """Remove Google SERP noise (buttons, pagination, PAA) from a snapshot."""
    lines = []
    for line in snap.split("\n"):
        # Skip lines containing noise patterns
        if any(noise in line for noise in _SERP_NOISE):
            continue
        # Skip generic "Read more" links (PAA expansions)
        if '"Read more"' in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def _mimo_pick_ref(snap: str, instruction: str) -> Optional[str]:
    """Ask MIMO to pick the best ref from a snapshot based on instruction."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("MIMO: OPENROUTER_API_KEY not set")
        return None
    prompt = (
        f"You are looking at a browser page snapshot. Each interactive element has a [ref=N] number.\n\n"
        f"SNAPSHOT:\n{snap[:8000]}\n\n"
        f"TASK: {instruction}\n\n"
        f"Do NOT list or repeat the elements. Reply with ONLY the ref number (e.g. '24'). If no matching element exists, reply 'none'."
    )
    try:
        model = os.getenv("MIMO_MODEL", "xiaomi/mimo-v2-omni")
        resp = http_requests.post(
            f"{os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 100000},
            timeout=60,
        ).json()
        if resp.get("error"):
            logger.warning("MIMO API error: %s", resp["error"])
            return None
        answer = (resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        logger.info("MIMO picked ref: %s", answer)
        if not answer or answer.lower() == "none":
            return None
        ref = re.search(r'\d+', answer)
        return ref.group(0) if ref else None
    except http_requests.exceptions.Timeout:
        logger.warning("MIMO call timed out")
        return None
    except Exception as e:
        logger.warning("MIMO call failed: %s", e)
        return None

def _mimo_pick_multiple_refs(snap: str, instruction: str, max_refs: int = 3) -> List[str]:
    """Ask MIMO to pick multiple refs from a snapshot. Returns list of ref strings."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return []
    prompt = (
        f"You are looking at a browser page snapshot. Each interactive element has a [ref=N] number.\n\n"
        f"SNAPSHOT:\n{snap[:8000]}\n\n"
        f"TASK: {instruction}\n\n"
        f"Do NOT list or repeat the elements. Reply with up to {max_refs} ref numbers, one per line (e.g. '12\\n5\\n23').\n"
        f"Order by priority (best first). If no matching elements exist, reply 'none'."
    )
    try:
        model = os.getenv("MIMO_MODEL", "xiaomi/mimo-v2-omni")
        resp = http_requests.post(
            f"{os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 100000},
            timeout=60,
        ).json()
        if resp.get("error"):
            logger.warning("MIMO API error: %s", resp["error"])
            return []
        answer = (resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        logger.info("MIMO picked refs: %s", answer.replace("\n", ", "))
        if not answer or answer.lower() == "none":
            return []
        refs = re.findall(r'\d+', answer)
        return refs[:max_refs]
    except http_requests.exceptions.Timeout:
        logger.warning("MIMO multi-ref call timed out")
        return []
    except Exception as e:
        logger.warning("MIMO multi-ref call failed: %s", e)
        return []


def _extract_page_images(subpage_label: str = "homepage") -> List[dict]:
    """Scroll current page and extract image URLs. Returns list of {url, source, subpage}."""
    # Force lazy-loaded images
    _eval("(function(){document.querySelectorAll('img').forEach(function(img){"
          "var ds=img.getAttribute('data-src')||img.getAttribute('data-lazy-src')||img.getAttribute('data-original');"
          "if((img.src.startsWith('data:')||!img.src)&&ds)img.src=ds;});})()")
    for _ in range(8):
        _eval("window.scrollBy(0,800)"); time.sleep(0.3)
    time.sleep(1)
    raw = _eval(
        "(function(){var imgs=document.querySelectorAll('img'),s=new Set(),u=[];"
        "for(var i=0;i<imgs.length;i++){var src=imgs[i].src||imgs[i].getAttribute('data-src')||'';"
        "if(!src||src.startsWith('data:')||src.length<30)continue;"
        "var sl=src.toLowerCase();"
        "if(['favicon','logo','icon','pixel','spacer','tracking','1x1','sprite','.svg','badge',"
        "'maps.googleapis','maps.gstatic','khms','tile.openstreetmap','map-tile']"
        ".some(function(k){return sl.includes(k);}))continue;"
        "if(!s.has(src)){s.add(src);u.push(src);}}"
        "return JSON.stringify({c:u.length,u:u});})()", timeout=30)
    entries = []
    try:
        for url in json.loads(raw).get("u", []):
            if not _is_junk(url):
                entries.append({"url": _boost_cdn(url), "source": "official_site", "subpage": subpage_label})
    except (json.JSONDecodeError, TypeError):
        pass
    return entries


def _snap_ref_info(snap: str, refs: List[str]) -> List[dict]:
    """Extract label and href for given refs from a snapshot string.

    Snapshot format (one per line): '- link "Hotel" [ref=3]'
    Returns list of {ref, label} dicts.
    """
    results = []
    for ref in refs:
        for line in snap.split("\n"):
            if f"[ref={ref}]" in line:
                m = re.search(r'"([^"]+)"', line)
                label = m.group(1).rstrip(".").split("|")[0].strip() if m else f"subpage_{ref}"
                results.append({"ref": ref, "label": label})
                break
        else:
            results.append({"ref": ref, "label": f"subpage_{ref}"})
    return results


def _extract_official_site(poi_name: str, domain: Optional[str] = None) -> List[dict]:
    """Extract photos from a hotel's official website using multi-turn navigation.

    Flow:
    1. Google search hotel name → snapshot search results → MIMO finds official website link
    2. Click into official site
    3. Snapshot homepage → LLM picks up to 3 subpages likely to have photos
    4. Visit each subpage, scroll, extract <img> URLs
    5. Aggregate and dedupe
    """
    # Step 1: Google search → snapshot → MIMO finds official website link
    encoded = poi_name.replace(" ", "+").replace("&", "%26")
    _open(f"https://www.google.com/search?q={encoded}")
    time.sleep(4)
    snap = _clean_serp_snapshot(_snap())

    ref = _mimo_pick_ref(
        snap,
        f"Find the official website link for '{poi_name}' on this Google search results page. "
        f"Look for the 'Website' button in the Knowledge Panel, or a link showing the hotel's "
        f"domain (e.g. marriott.com, hilton.com, hyatt.com, or the hotel's own domain). "
        f"Do NOT pick Google links, 'More results', map links, booking sites (expedia, booking.com), "
        f"or review sites (tripadvisor, yelp). Pick the direct hotel website link.",
    )
    if ref:
        cmd("click", ref=ref); time.sleep(5)
        domain = _eval("window.location.hostname") or ""
        domain = domain.replace("www.", "")
        logger.info("Official site: navigated to %s (from search results ref=%s)", domain, ref)
    else:
        # Fallback: try Gemini domain resolution if MIMO can't find the link
        logger.info("Official site: MIMO found no website link in search results, trying Gemini fallback")
        _close()
        if not domain:
            try:
                from scraping.core.scraper import resolve_poi_domain
                domain = resolve_poi_domain(poi_name)
            except Exception:
                pass
        if not domain:
            logger.info("Official site: no domain found for '%s'", poi_name)
            return []
        _open(f"https://www.{domain}"); time.sleep(6)

    # Dismiss cookie banners if present
    _eval("(function(){var b=document.querySelector('[class*=cookie] button[class*=accept],"
          "[id*=cookie] button,[class*=consent] button:last-child');"
          "if(b)b.click();})()")
    time.sleep(1)

    # Step 2: Snapshot homepage → LLM picks top 3 photo-rich subpages
    snap = _snap()
    logger.info("Official site: homepage snapshot %d chars, first link: %s",
                len(snap), snap[:80].strip())
    # Retry once if MIMO returns empty (rate-limit or transient failure)
    subpage_refs = []
    for attempt in range(2):
        if attempt > 0:
            time.sleep(3)
            logger.info("Official site: retrying subpage selection (attempt %d)", attempt + 1)
        subpage_refs = _mimo_pick_multiple_refs(
            snap,
            f"This is the homepage of {poi_name}, a luxury hotel. "
            f"Pick up to 3 navigation links most likely to show high-quality hotel photos. "
            f"Good targets: Rooms, Suites, Dining, Restaurant, Spa, Pool, Gallery, Photos, Amenities, Experiences. "
            f"Bad targets: Book Now, Contact, Careers, Login, FAQ, Privacy, Terms. "
            f"Return the ref numbers of your top 3 picks, best first.",
            max_refs=3,
        )
        if subpage_refs:
            break

    # Get labels for the selected refs
    ref_info = _snap_ref_info(snap, subpage_refs) if subpage_refs else []
    logger.info("Official site: homepage analyzed, %d subpages to visit: %s",
                len(ref_info), [r["label"] for r in ref_info])

    # Resolve hrefs for selected subpages via JS (refs are ephemeral, hrefs are stable)
    labels = [info["label"] for info in ref_info]
    subpage_urls = []
    if labels:
        labels_js = json.dumps(labels)
        raw_urls = _eval(
            f"(function(){{var labels={labels_js},result=[];"
            f"var links=document.querySelectorAll('a');"
            f"for(var i=0;i<labels.length;i++){{"
            f"var lbl=labels[i].toLowerCase();"
            f"for(var j=0;j<links.length;j++){{"
            f"if(links[j].textContent.trim().toLowerCase()===lbl&&links[j].href){{"
            f"result.push({{label:labels[i],url:links[j].href}});break;}}}}}}"
            f"return JSON.stringify(result);}})()")
        try:
            subpage_urls = json.loads(raw_urls) if raw_urls else []
        except (json.JSONDecodeError, TypeError):
            pass
        for sp in subpage_urls:
            logger.info("Official site: '%s' → %s", sp["label"], sp["url"][:80])

    # Step 3: Extract images from homepage first
    all_entries = _extract_page_images("homepage")
    logger.info("Official site: %d images from homepage", len(all_entries))

    # Step 4: Visit each subpage by URL
    seen_urls = {e["url"] for e in all_entries}
    for sp in subpage_urls:
        label, url = sp["label"], sp["url"]
        try:
            _eval(f"window.location.href='{url}'"); time.sleep(5)
            _eval("window.scrollTo(0,0)"); time.sleep(1)
            page_entries = _extract_page_images(label)
            new_entries = [e for e in page_entries if e["url"] not in seen_urls]
            for e in new_entries:
                seen_urls.add(e["url"])
            all_entries.extend(new_entries)
            logger.info("Official site: %d new images from '%s' (%s)",
                        len(new_entries), label, (_eval("window.location.pathname") or "?")[:50])
        except Exception as exc:
            logger.warning("Official site: failed on subpage '%s': %s", label, exc)

    _close()

    # Apply global limit
    if len(all_entries) > OFFICIAL_SITE_MAX:
        all_entries = all_entries[:OFFICIAL_SITE_MAX]

    logger.info("Official site: %d total URLs from '%s'", len(all_entries), domain)
    return all_entries

def bb_extract_urls(poi_name: str, output_path: str) -> str:
    """Extract image URLs from all 3 sources. Saves JSON manifest."""
    _check_connection()
    logger.info("bb-browser URL extraction: %s", poi_name)
    all_urls = []
    from scraping.core.kp_extractor import extract_kp_tabs
    for label, fn in [("KP", extract_kp_tabs), ("GImg", _extract_google_images), ("Official", _extract_official_site)]:
        try:
            all_urls.extend(fn(poi_name))
        except Exception as e:
            logger.warning("%s failed: %s", label, e)
    seen, unique = set(), []
    for e in all_urls:
        if e["url"] not in seen:
            seen.add(e["url"]); unique.append(e)
    manifest = {"poi_name": poi_name, "extracted_at": datetime.now(timezone.utc).isoformat(),
                "source_counts": {s: sum(1 for e in unique if e["source"] == s) for s in ("google_kp", "google_images", "official_site")},
                "total": len(unique), "urls": unique}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest: %s (%d URLs: KP=%d, GImg=%d, Off=%d)", output_path, manifest["total"],
                manifest["source_counts"]["google_kp"], manifest["source_counts"]["google_images"], manifest["source_counts"]["official_site"])
    return output_path

def _download_with_retry(url: str, dest: str) -> bool:
    parsed = urlparse(url)
    ref = f"{parsed.scheme}://{parsed.netloc}/"
    for attempt, hdrs in enumerate([{**_DL_HEADERS, "Referer": ref}, _DL_HEADERS]):
        try:
            resp = http_requests.get(url, timeout=_DL_TIMEOUT, headers=hdrs, stream=True)
            if resp.status_code == 403 and attempt == 0:
                continue
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "image" not in ct and "octet-stream" not in ct:
                return False
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            if os.path.getsize(dest) < 5000:
                os.remove(dest); return False
            return True
        except http_requests.exceptions.HTTPError:
            if attempt == 0: continue
            return False
        except Exception:
            return False
    return False

def bb_download_from_manifest(manifest_path: str, output_dir: str, max_images: int = None) -> List[str]:
    """Download images from a URL manifest. No bb-browser needed."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    urls = manifest.get("urls", [])
    poi_name = manifest.get("poi_name", "unknown")
    if max_images is not None: urls = urls[:max_images]
    os.makedirs(output_dir, exist_ok=True)
    start_idx = len([f for f in Path(output_dir).iterdir() if f.is_file() and f.name.startswith("candidate_")])
    downloaded, failed = [], 0
    logger.info("Downloading %d URLs for '%s'...", len(urls), poi_name)
    for i, entry in enumerate(urls):
        url = entry["url"]
        ext = ".jpg"
        for e in (".png", ".webp", ".jpeg"):
            if e in url.lower(): ext = e; break
        dest = os.path.join(output_dir, f"candidate_{start_idx + i:04d}{ext}")
        if _download_with_retry(url, dest):
            downloaded.append(dest)
        else:
            failed += 1
    logger.info("Downloaded %d/%d for '%s' (%d failed)", len(downloaded), len(urls), poi_name, failed)
    return sorted(str(f) for f in Path(output_dir).iterdir() if f.is_file() and f.suffix.lower() in _IMG_EXTS)
