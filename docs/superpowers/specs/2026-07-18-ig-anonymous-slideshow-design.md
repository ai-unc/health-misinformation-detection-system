# Instagram Slideshow: Anonymous-First Ingestion — Design

**Date:** 2026-07-18 · **Status:** Approved · **Branch:** `feat/ig-anonymous-slideshow`

## Problem

Instagram `/p/` slideshow posts currently hard-require `MAVEN_IG_COOKIES` (a
Netscape cookies.txt exported from a logged-in Instagram session). Without it,
`download_slideshow()` refuses before any network call. The cookie file is the
whole accessibility problem: it needs a personal Instagram account, a manual
export step, `chmod 600` handling of a live session credential, and it expires
silently. gallery-dl cannot help anonymously — its Instagram extractor is
redirected to the login page (reproduced 2026-07-18).

## Validated solution (empirical, 2026-07-18)

`GET https://www.instagram.com/p/<shortcode>/embed/captioned/` — the public
surface Instagram serves to any third-party blog embed — returns HTTP 200
**with no cookies** when fetched with real-Chrome TLS impersonation
(`curl_cffi`, `impersonate='chrome'`). The page contains a
`"contextJSON":"…"` JSON-string whose decoded value holds
`gql_data.shortcode_media`:

- full caption at `edge_media_to_caption.edges[0].node.text`
- owner at `owner.username`
- every carousel slide's `display_url` in carousel order at
  `edge_sidecar_to_children.edges[].node.display_url`
- `__typename`: `GraphSidecar` (carousel) / `GraphImage` (single) /
  `GraphVideo` (video post)

The `display_url` images (signed fbcdn URLs) also download anonymously at
full resolution. Verified on two accounts: `DRzdgElEf3N` (12 slides, 599-char
caption — identical coverage to the 2026-07-16 cookie-authenticated live run)
and `DVhV_kzGnOf` (7 slides). Plain curl without TLS impersonation gets a
decoy `httpErrorPage` shell — impersonation is the load-bearing ingredient.
The anonymous GraphQL `doc_id` endpoint is dead (403). `curl_cffi` is already
pinned in `maven_app/requirements.txt` (yt-dlp dependency): **zero new
dependencies**.

## Architecture

New module `maven_app/instagram_embed.py` (transport + parse only; all
user-facing message text stays in `video_source.py`):

- `download_slideshow_anonymous(url, tmp_dir) -> (List[Path], dict)` — same
  contract as `download_slideshow()`. Fetches the embed page, parses
  `contextJSON`, downloads each slide to `tmp_dir/embed/NN.jpg` (zero-padded,
  order preserved by construction), returns
  `({'description': caption, 'username': owner})` metadata — the exact
  Instagram schema `slideshow.py`'s normalizers already read. **No changes to
  `slideshow.py`.**
- Exceptions:
  - `EmbedUnavailableError(RuntimeError)` — anonymous path can't serve this
    post (non-200, missing/unparseable contextJSON, null `gql_data`, image
    download failure). Fallback-eligible.
  - `NotASlideshowError(RuntimeError)` — post is definitively a `GraphVideo`.
    Terminal; no fallback (cookies wouldn't make it a slideshow).
- Small seams for tests: `_fetch_embed_page(shortcode) -> str` and
  `_download_image(url, dest)`.
- Parse: regex `"contextJSON":"((?:\\.|[^"\\])*)"`, unescape via
  `json.loads('"…"')`, decode inner JSON, read `gql_data.shortcode_media`.
  Mixed carousels degrade gracefully: `GraphVideo` children contribute their
  poster-frame `display_url`, consistent with text-mode semantics.

`video_source.download_slideshow()` becomes tiered for Instagram URLs:

1. **Anonymous first** (lazy `import instagram_embed`; `ImportError` treated
   as unavailable).
2. On `EmbedUnavailableError`: if `MAVEN_IG_COOKIES` is set → existing
   gallery-dl + cookies leg (hoisted unchanged into
   `_download_slideshow_gallery_dl()`); else raise the reworded
   `INSTAGRAM_COOKIE_MESSAGE`.
3. On `NotASlideshowError`: raise the existing "No images found… may be a
   video post" message.

TikTok is untouched (still goes straight to the gallery-dl leg, no cookies).
The gallery-dl leg writes to `tmp_dir` root and globs non-recursively, so the
`embed/` subdirectory keeps a failed anonymous attempt's partial files out of
the fallback's results.

Reworded `INSTAGRAM_COOKIE_MESSAGE` (cookies are now the fallback, not a
requirement): "Instagram blocked anonymous access to this post — it may be
private, removed, or rate-limited. Retry later, or set MAVEN_IG_COOKIES to a
logged-in cookies.txt for instagram.com to access it with your account."

## Testing

- Fixture-based unit tests (no network): fixture HTML built in-test by
  double-`json.dumps` so escaping matches the real page shape. Cover: carousel
  parse (caption/owner/ordered URLs), single `GraphImage`, `GraphVideo` →
  `NotASlideshowError`, missing contextJSON → `EmbedUnavailableError`.
- Tiering tests: anonymous success → gallery-dl never invoked; anonymous
  failure + no cookies → cookie message; + cookies → gallery-dl leg with
  `--cookies` (existing command assertions preserved); `NotASlideshowError` →
  video-post message even with cookies set.
- Existing tests 4/5B/5C gain a patch making the anonymous path raise, so
  they keep exercising the gallery-dl leg.
- `--live` Instagram test flipped to first-class: explicitly unsets
  `MAVEN_IG_COOKIES` and asserts full caption + slide extraction anonymously.

## Docs

CLAUDE.md slideshow section: Instagram slideshows work anonymously out of the
box; `MAVEN_IG_COOKIES` demoted to optional fallback (private/restricted
posts, rate-limit relief; still passed to yt-dlp for Reels). Record the
2026-07-18 anonymous live verification alongside the 2026-07-16 cookie one.

## Risks

Unofficial surface: Meta reshaped the embed page once before (server-rendered
HTML → JS shell with inline contextJSON). Mitigation: the tiered design
degrades to exactly today's behavior on any failure — never worse. Usage fits
the endpoint's intent: single user-initiated public posts at trivial volume
for public-health research; private content stays behind the operator's own
login.
