"""KP (Knowledge Panel) image extraction using bb-browser API.

Uses bb-browser's snapshot + ref workflow for navigation:
- snapshot -i → find Photos button by text, not hardcoded CSS class
- cmd("click", ref=N) → click tabs
- cmd("scroll", direction="down") → scroll inside KP panel
- _eval() → extract image URLs from DOM (snapshot doesn't expose img src)

Drop-in replacement for _extract_kp_tabs in bb_source.py.
"""

import json
import logging
import re
import time
from typing import List

from scraping.core.bb_source import cmd, _open, _snap, _eval, _close
from scraping.config import KP_ALL_TAB_QUOTA, KP_OTHER_TAB_QUOTA, KP_TOTAL_CAP

logger = logging.getLogger(__name__)


def _find_photos_button_ref(snap: str) -> str | None:
    """Find the KP 'Photos' button ref from a snapshot.

    Matches patterns like:
      - button "Photos" [ref=5]
      - button "See photos" [ref=5]
      - link "See photos" [ref=5]  (side KP variant)
    """
    for line in snap.split("\n"):
        if re.search(r'(?:button|link).*"[^"]*(?:[Ss]ee\s+)?[Pp]hotos[^"]*".*\[ref=(\d+)\]', line):
            m = re.search(r'\[ref=(\d+)\]', line)
            if m:
                return m.group(1)
    return None


def _find_hotels_pack_name(snap: str, poi_name: str = "") -> str | None:
    """Extract best-matching hotel name from a Hotels pack in the snapshot.

    Hotels pack entries look like: link "W Miami $444 4.3 out of 5..." [ref=30]
    If poi_name is provided, picks the entry whose name overlaps most with it.
    Otherwise returns the first match.
    """
    candidates = []
    for line in snap.split("\n"):
        m = re.search(r'link\s+"([^"]+?)\s*\$\d+', line)
        if m:
            hotel_name = m.group(1).strip()
            if hotel_name:
                candidates.append(hotel_name)
    if not candidates:
        return None
    if not poi_name:
        return candidates[0]
    # Score by word overlap with poi_name
    poi_words = set(poi_name.lower().split())
    best, best_score = candidates[0], 0
    for c in candidates:
        score = len(poi_words & set(c.lower().split()))
        if score > best_score:
            best, best_score = c, score
    return best


def _parse_tab_refs(snap: str) -> dict:
    """Parse KP tab names and their refs from a snapshot.

    Matches: "Change collection to Rooms" [ref=12]
    Returns: {"Rooms": "12", "Exterior": "13", ...}
    """
    refs = {}
    for line in snap.split("\n"):
        m = re.search(r'"Change collection to (.+?)" \[ref=(\d+)\]', line.strip())
        if m:
            refs[m.group(1)] = m.group(2)
    return refs


def _extract_images_from_page() -> List[str]:
    """Extract googleusercontent image URLs from current page DOM via eval."""
    raw = _eval(
        "(function(){var imgs=document.querySelectorAll(\"img[src*='googleusercontent']\"),"
        "seen=new Set(),urls=[];"
        "for(var i=0;i<imgs.length;i++){var s=imgs[i].src;"
        "if(s.includes('/a/')||s.includes('/a-/')||imgs[i].naturalWidth<30)continue;"
        "var b=s.replace(/=.*$/,'');"
        "if(!seen.has(b)&&b.length>50){seen.add(b);urls.push(b);}}"
        "return JSON.stringify({c:urls.length,u:urls});})()"
    )
    try:
        return json.loads(raw).get("u", [])
    except (json.JSONDecodeError, TypeError):
        return []


def _scroll_kp_panel(rounds: int = 15, pause: float = 0.3):
    """Scroll inside the KP photo panel's scrollable container.

    Dynamically finds the scrollable div that contains googleusercontent images,
    rather than hardcoding a CSS class (which Google can change anytime).
    Falls back to bb-browser page scroll if no scrollable container found.
    """
    # Find the scrollable container with KP images inside it
    _eval(
        "(function(){"
        "var all=document.querySelectorAll('*');"
        "for(var i=0;i<all.length;i++){"
        "var el=all[i],s=getComputedStyle(el);"
        "if((s.overflowY==='auto'||s.overflowY==='scroll')&&el.scrollHeight>el.clientHeight+100"
        "&&el.querySelectorAll(\"img[src*='googleusercontent']\").length>0){"
        "window.__kpScrollTarget=el;return;}}"
        "})()"
    )
    for _ in range(rounds):
        scrolled = _eval(
            "(function(){"
            "var el=window.__kpScrollTarget;"
            "if(el){el.scrollTop+=400;return 'ok';}"
            "return 'no_target';})()"
        )
        if scrolled == "no_target":
            # Fallback: page-level scroll
            cmd("scroll", direction="down", amount=400)
        time.sleep(pause)
    time.sleep(1)


def extract_kp_tabs(poi_name: str) -> List[dict]:
    """Extract images from Google Knowledge Panel photo tabs.

    Flow (per bb-browser skill):
    1. open Google search
    2. snapshot -i → find Photos button by text
    3. click Photos ref → KP panel opens
    4. snapshot -i → parse tab refs
    5. For each tab: click ref → scroll → eval extract images → re-snapshot
    6. close

    Returns list of {"url": ..., "source": "google_kp", "category": "Rooms"}.
    """
    encoded = poi_name.replace(' ', '+').replace('&', '%26')
    _open(f"https://www.google.com/search?q={encoded}")
    time.sleep(4)

    # Step 1: Find and click Photos button via snapshot
    snap = _snap()
    photos_ref = _find_photos_button_ref(snap)
    if not photos_ref:
        # Fallback: check for Hotels pack (multi-hotel list) and re-search
        pack_name = _find_hotels_pack_name(snap, poi_name)
        if pack_name and pack_name.lower() != poi_name.lower():
            logger.info("KP: no Photos button — found Hotels pack, re-searching '%s'", pack_name)
            _close()
            encoded = pack_name.replace(' ', '+').replace('&', '%26')
            _open(f"https://www.google.com/search?q={encoded}")
            time.sleep(4)
            snap = _snap()
            photos_ref = _find_photos_button_ref(snap)
        if not photos_ref:
            logger.warning("KP: no Photos button found for '%s'", poi_name)
            _close()
            return []

    cmd("click", ref=photos_ref)
    time.sleep(4)

    # Step 2: Snapshot to get tab list
    snap = _snap()
    if not snap:
        _close()
        return []

    tab_names = list(_parse_tab_refs(snap).keys())
    if not tab_names:
        logger.warning("KP: no tabs for '%s'", poi_name)
        _close()
        return []

    logger.info("KP: %d tabs: %s", len(tab_names), tab_names)

    # Step 3: Visit each tab, scroll, extract images
    entries, seen = [], set()
    for tab_name in tab_names:
        # Re-snapshot for fresh refs each iteration
        snap = _snap()
        refs = _parse_tab_refs(snap)
        ref = refs.get(tab_name)
        if not ref:
            continue

        cmd("click", ref=ref)
        time.sleep(3)

        # Scroll inside KP panel to load lazy images
        _scroll_kp_panel(rounds=15, pause=0.3)

        # Extract image URLs via eval
        urls = _extract_images_from_page()
        tab_quota = KP_ALL_TAB_QUOTA if tab_name.lower() == "all" else KP_OTHER_TAB_QUOTA
        n = 0
        for url in urls:
            if n >= tab_quota:
                break
            if len(entries) >= KP_TOTAL_CAP:
                logger.info("KP: total cap %d reached, stopping", KP_TOTAL_CAP)
                break
            full = url + "=s0"
            if full not in seen:
                seen.add(full)
                entries.append({"url": full, "source": "google_kp", "category": tab_name})
                n += 1

        logger.info("KP: '%s' -> %d (quota %d)", tab_name, n, tab_quota)

        if len(entries) >= KP_TOTAL_CAP:
            break

    _close()
    logger.info("KP: %d URLs for '%s'", len(entries), poi_name)
    return entries
