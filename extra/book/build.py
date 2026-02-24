#!/usr/bin/env python3
"""Build script for the tinygrad internals book.

Generates HTML (multi-page with navigation) and EPUB from markdown sources.
Only requires markdown-it-py and pygments (both already in tinygrad's venv).

Usage:
  python build.py html          # build HTML
  python build.py epub          # build EPUB
  python build.py all           # build both (default)
  python build.py serve         # build HTML + start local server
  python build.py clean         # remove build/
  python build.py html --lang zh  # build Chinese version
"""
import sys, os, re, shutil, zipfile, uuid, http.server, functools
from pathlib import Path
from datetime import datetime

BOOK_DIR = Path(__file__).parent.resolve()

# chapter order — derived from SUMMARY.md
def get_chapters(lang: str = "en") -> list[tuple[str, str, Path]]:
  """Return [(part_title, chapter_title, path), ...]"""
  src = BOOK_DIR / "zh" if lang == "zh" else BOOK_DIR
  summary = src / "SUMMARY.md"
  if not summary.exists():
    # fallback: glob numbered files
    files = sorted(src.glob("[0-9][0-9]_*.md"))
    return [("", f.stem.split("_", 1)[1].replace("_", " ").title(), f) for f in files]

  chapters: list[tuple[str, str, Path]] = []
  current_part = ""
  for line in summary.read_text().splitlines():
    line = line.strip()
    if line.startswith("# "):
      current_part = line[2:].strip()
    m = re.match(r'-\s+\[(.+?)\]\((.+?)\)', line)
    if m:
      title, fname = m.group(1), m.group(2)
      p = src / fname
      if p.exists():
        chapters.append((current_part, title, p))
  return chapters

# ── Markdown rendering ─────────────────────────���────────────────────────────

def render_md(text: str) -> str:
  from markdown_it import MarkdownIt
  from markdown_it.rules_block import StateBlock  # noqa: F401
  md = MarkdownIt("gfm-like", {"html": True, "typographer": False, "linkify": False})
  md.disable("linkify")
  # add syntax highlighting
  default_fence = md.renderer.rules.get("fence")
  def fence_with_highlight(tokens, idx, options, env):
    token = tokens[idx]
    lang = token.info.strip().split()[0] if token.info else ""
    code = token.content
    if lang:
      try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import HtmlFormatter
        lexer = get_lexer_by_name(lang, stripall=True)
        return highlight(code, lexer, HtmlFormatter(nowrap=False, cssclass="highlight"))
      except Exception:
        pass
    escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    cls = f' class="language-{lang}"' if lang else ""
    return f'<pre><code{cls}>{escaped}</code></pre>\n'
  md.renderer.rules["fence"] = fence_with_highlight
  return md.render(text)

# ── HTML builder ─────────────────────────────────────────────────────────────

CSS = """
:root { --bg: #fff; --fg: #1a1a1a; --code-bg: #f6f8fa; --border: #d0d7de;
        --link: #0969da; --nav-bg: #f6f8fa; --accent: #0550ae; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --code-bg: #161b22; --border: #30363d;
          --link: #58a6ff; --nav-bg: #161b22; --accent: #79c0ff; }
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       line-height: 1.6; color: var(--fg); background: var(--bg); max-width: 52em;
       margin: 0 auto; padding: 1em 2em; }
h1 { border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
h2 { border-bottom: 1px solid var(--border); padding-bottom: 0.2em; margin-top: 1.5em; }
h3 { margin-top: 1.3em; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px;
      padding: 1em; overflow-x: auto; font-size: 0.875em; line-height: 1.45; }
code { font-family: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, Consolas, monospace;
       font-size: 0.875em; }
:not(pre) > code { background: var(--code-bg); padding: 0.2em 0.4em; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid var(--border); padding: 0.5em 0.75em; text-align: left; }
th { background: var(--nav-bg); font-weight: 600; }
tr:nth-child(even) { background: var(--nav-bg); }
blockquote { border-left: 4px solid var(--accent); margin: 1em 0; padding: 0.5em 1em;
             background: var(--nav-bg); }
.nav { display: flex; justify-content: space-between; padding: 1em 0;
       border-top: 1px solid var(--border); margin-top: 2em; font-size: 0.9em; }
.nav a { padding: 0.3em 0.8em; background: var(--nav-bg); border-radius: 4px;
         border: 1px solid var(--border); }
.toc { background: var(--nav-bg); border: 1px solid var(--border); border-radius: 6px;
       padding: 1em 1.5em; margin: 1em 0; }
.toc ul { list-style: none; padding-left: 1.2em; }
.toc > ul { padding-left: 0; }
.toc li { margin: 0.2em 0; }
.part-title { font-weight: 700; margin-top: 0.8em; color: var(--accent); }
.highlight { background: var(--code-bg); border-radius: 6px; padding: 1em;
             overflow-x: auto; border: 1px solid var(--border); }
.highlight pre { border: none; padding: 0; margin: 0; background: transparent; }
"""

def build_html(lang: str = "en"):
  chapters = get_chapters(lang)
  if not chapters:
    print("No chapters found."); return
  out_dir = BOOK_DIR / "build" / ("html-zh" if lang == "zh" else "html")
  out_dir.mkdir(parents=True, exist_ok=True)

  # pygments CSS
  try:
    from pygments.formatters import HtmlFormatter
    pygments_css = HtmlFormatter().get_style_defs('.highlight')
  except Exception:
    pygments_css = ""

  full_css = CSS + "\n" + pygments_css

  def page(title, body, prev_link=None, next_link=None, next_title=None, prev_title=None):
    nav = '<div class="nav">'
    nav += f'<a href="{prev_link}">&#8592; {prev_title}</a>' if prev_link else '<span></span>'
    nav += '<a href="index.html">Home</a>'
    nav += f'<a href="{next_link}">{next_title} &#8594;</a>' if next_link else '<span></span>'
    nav += '</div>'
    return f"""<!DOCTYPE html>
<html lang="{"zh-CN" if lang == "zh" else "en"}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{full_css}</style></head>
<body>{body}\n{nav}</body></html>"""

  # build index
  toc_html = '<div class="toc"><ul>'
  current_part = None
  for part, title, path in chapters:
    if part and part != current_part:
      current_part = part
      toc_html += f'<li class="part-title">{part}</li>'
    slug = path.stem
    toc_html += f'<li><a href="{slug}.html">{title}</a></li>'
  toc_html += '</ul></div>'

  readme = (BOOK_DIR / "zh" / "README.md") if lang == "zh" else (BOOK_DIR / "README.md")
  index_body = render_md(readme.read_text()) if readme.exists() else "<h1>Tinygrad Internals</h1>"
  (out_dir / "index.html").write_text(page("Tinygrad Internals", index_body))

  # build each chapter
  filenames = [(p.stem, title) for _, title, p in chapters]
  for i, (part, title, path) in enumerate(chapters):
    body = render_md(path.read_text())
    prev_link = f"{filenames[i-1][0]}.html" if i > 0 else "index.html"
    prev_title = filenames[i-1][1] if i > 0 else "Home"
    next_link = f"{filenames[i+1][0]}.html" if i < len(chapters) - 1 else None
    next_title = filenames[i+1][1] if i < len(chapters) - 1 else None
    html = page(title, body, prev_link, next_link, next_title, prev_title)
    (out_dir / f"{path.stem}.html").write_text(html)

  print(f"HTML built: {out_dir}/ ({len(chapters)} chapters)")

# ── EPUB builder ─────────────────────────────────────────────────────────────

def build_epub(lang: str = "en"):
  chapters = get_chapters(lang)
  if not chapters:
    print("No chapters found."); return
  out_dir = BOOK_DIR / "build"
  out_dir.mkdir(parents=True, exist_ok=True)
  epub_name = "tinygrad-internals-zh.epub" if lang == "zh" else "tinygrad-internals.epub"
  epub_path = out_dir / epub_name

  book_id = str(uuid.uuid4())
  title = "Tinygrad 内部原理" if lang == "zh" else "Tinygrad Internals"
  book_lang = "zh-CN" if lang == "zh" else "en"
  date = datetime.now().strftime("%Y-%m-%d")

  try:
    from pygments.formatters import HtmlFormatter
    pygments_css = HtmlFormatter().get_style_defs('.highlight')
  except Exception:
    pygments_css = ""

  style_css = """
body { font-family: serif; line-height: 1.6; margin: 1em; }
h1 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
h2 { margin-top: 1.5em; }
pre { background: #f6f8fa; border: 1px solid #ddd; border-radius: 4px;
      padding: 0.8em; overflow-x: auto; font-size: 0.85em; }
code { font-family: monospace; font-size: 0.9em; }
:not(pre) > code { background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.6em; }
th { background: #f0f0f0; }
.highlight pre { border: none; padding: 0; background: transparent; }
""" + pygments_css

  def xhtml_wrap(ch_title, body):
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{book_lang}">
<head><meta charset="utf-8"/><title>{ch_title}</title>
<link rel="stylesheet" type="text/css" href="style.css"/></head>
<body>{body}</body></html>"""

  with zipfile.ZipFile(epub_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # mimetype must be first and uncompressed
    zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

    # container.xml
    zf.writestr("META-INF/container.xml", """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>""")

    # style
    zf.writestr("OEBPS/style.css", style_css)

    # chapters
    manifest_items = ['<item id="style" href="style.css" media-type="text/css"/>']
    spine_items = []
    toc_entries = []

    for i, (part, ch_title, path) in enumerate(chapters):
      ch_id = f"ch{i:02d}"
      fname = f"{ch_id}.xhtml"
      body = render_md(path.read_text())
      # fix self-closing tags for XHTML
      body = re.sub(r'<(img|br|hr)([^>]*?)(?<!/)>', r'<\1\2/>', body)
      zf.writestr(f"OEBPS/{fname}", xhtml_wrap(ch_title, body))
      manifest_items.append(f'<item id="{ch_id}" href="{fname}" media-type="application/xhtml+xml"/>')
      spine_items.append(f'<itemref idref="{ch_id}"/>')
      toc_entries.append((ch_id, fname, ch_title, part))

    # toc.xhtml
    toc_body = f"<h1>{title}</h1><nav epub:type=\"toc\"><ol>"
    current_part = None
    for ch_id, fname, ch_title, part in toc_entries:
      if part and part != current_part:
        if current_part is not None:
          toc_body += "</ol></li>"
        current_part = part
        toc_body += f"<li><span>{part}</span><ol>"
      toc_body += f'<li><a href="{fname}">{ch_title}</a></li>'
    if current_part is not None:
      toc_body += "</ol></li>"
    toc_body += "</ol></nav>"
    toc_xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{book_lang}">
<head><meta charset="utf-8"/><title>Table of Contents</title></head>
<body>{toc_body}</body></html>"""
    zf.writestr("OEBPS/toc.xhtml", toc_xhtml)
    manifest_items.append('<item id="toc" href="toc.xhtml" media-type="application/xhtml+xml" properties="nav"/>')

    # content.opf
    manifest = "\n    ".join(manifest_items)
    spine = "\n    ".join(spine_items)
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:uuid:{book_id}</dc:identifier>
    <dc:title>{title}</dc:title>
    <dc:language>{book_lang}</dc:language>
    <dc:creator>tinygrad contributors</dc:creator>
    <dc:date>{date}</dc:date>
    <meta property="dcterms:modified">{date}T00:00:00Z</meta>
  </metadata>
  <manifest>
    {manifest}
  </manifest>
  <spine>
    {spine}
  </spine>
</package>"""
    zf.writestr("OEBPS/content.opf", opf)

  print(f"EPUB built: {epub_path} ({len(chapters)} chapters)")

# ── serve ────────────────────────────────────────────────────────────────────

def serve(lang: str = "en", port: int = 8000):
  build_html(lang)
  out_dir = BOOK_DIR / "build" / ("html-zh" if lang == "zh" else "html")
  os.chdir(out_dir)
  handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
  print(f"Serving at http://localhost:{port}")
  http.server.HTTPServer(("", port), handler).serve_forever()

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
  args = sys.argv[1:]
  lang = "zh" if "--lang" in args and "zh" in args else "en"
  args = [a for a in args if a not in ("--lang", "zh", "en")]
  cmd = args[0] if args else "all"

  if cmd == "html":
    build_html(lang)
  elif cmd == "epub":
    build_epub(lang)
  elif cmd == "all":
    build_html(lang)
    build_epub(lang)
  elif cmd == "serve":
    serve(lang)
  elif cmd == "clean":
    shutil.rmtree(BOOK_DIR / "build", ignore_errors=True)
    print("Cleaned build/")
  else:
    print(__doc__)
    sys.exit(1)
