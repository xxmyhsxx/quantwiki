"""Read-only Wiki structure, Markdown, source identity and navigation checks.

Uses the documented scalar and string-list frontmatter subset; no third-party
packages. This checks mechanical consistency, not scientific claims or runtime
behavior. Missing raw is reported as skipped unless --require-raw is supplied.
"""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from urllib.parse import unquote

CATEGORIES = {'fundamentals', 'theory', 'methods', 'research', 'implementation'}
TYPES = {'source', 'concept', 'method', 'implementation', 'comparison', 'recipe', 'question', 'index'}
PINNED_CODE = re.compile(r'https://github\.com/[^/]+/[^/]+/(?:blob|tree)/[0-9a-f]{40}(?:/[^\s]+)?$')
ARXIV = re.compile(r'(?:arXiv:\s*|arxiv\.org/(?:abs|pdf|src)/)(\d{4}\.\d{4,5})')


def prose(text):
    text = re.sub(r'^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$', '', text, flags=re.M | re.S)
    return re.sub(r'(`+).*?\1', '', text)


def anchors(text):
    counts = Counter()
    out = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    for h in re.findall(r'^#{1,6}\s+(.+?)\s*#*\s*$', prose(text), re.M):
        h = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', h)
        base = re.sub(r'[^\w\-\s]', '', h.lower()).replace(' ', '-')
        out.add(base + ('-' + str(counts[base]) if counts[base] else ''))
        counts[base] += 1
    return out


def metadata(text):
    if not text.startswith('---\n'):
        return {}, text, ['missing frontmatter']
    parts = text.split('---', 2)
    if len(parts) != 3:
        return {}, text, ['unclosed frontmatter']
    data, errors, current = {}, [], None
    for line in parts[1].splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        item = re.match(r'^  - (.+)$', line)
        if item:
            if current is None or not isinstance(data[current], list):
                errors.append('list item without a list field')
            else:
                data[current].append(item[1].strip().strip('"\''))
            continue
        m = re.match(r'^([a-z_]+):\s*(.*?)\s*$', line)
        if not m:
            errors.append('unsupported frontmatter syntax: ' + line)
            continue
        key, val = m.groups()
        if key in data:
            errors.append('duplicate metadata field: ' + key)
        if val in ('', '[]'):
            data[key] = []
        else:
            data[key] = val.strip('"\'')
        current = key
    return data, parts[2], errors


def check(root, require_raw=False):
    root = Path(root).resolve()
    wiki = root / 'wiki'
    errors, warnings, skipped = [], [], []
    counts = Counter()
    pages, bodies, graph, source_cache = {}, {}, {}, {}
    raw_present = (root / 'raw').is_dir()
    if not raw_present:
        (errors if require_raw else skipped).append('raw absent: local source files and raw metadata not verified')

    def fail(path, message):
        errors.append(str(path.relative_to(root)) + ': ' + message)

    for name in ('README.md', 'INDEX.md', 'LOG.md'):
        if not (wiki / name).is_file():
            errors.append('missing wiki/' + name)
    for child in wiki.iterdir() if wiki.is_dir() else []:
        if child.is_dir() and child.name not in CATEGORIES | {'assets'}:
            errors.append('unexpected Wiki directory: ' + child.name)
    for path in sorted(wiki.rglob('*.md')):
        if 'assets' in path.relative_to(wiki).parts:
            continue
        text = path.read_text()
        if path.parent == wiki and path.name in {'README.md', 'LOG.md'}:
            bodies[path] = text
            continue
        meta, body, issues = metadata(text)
        for issue in issues:
            fail(path, issue)
        bodies[path] = body
        is_index = path == wiki / 'INDEX.md'
        for key in ('title', 'type', 'sources') + (() if is_index else ('tags',)):
            if key not in meta:
                fail(path, 'missing field: ' + key)
        if not isinstance(meta.get('title'), str) or not meta.get('title', '').strip():
            fail(path, 'title must be a nonempty string')
        if meta.get('type') not in TYPES:
            fail(path, 'invalid page type')
        if is_index and meta.get('type') != 'index':
            fail(path, 'INDEX must have type: index')
        if not is_index:
            if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', path.stem):
                fail(path, 'invalid filename')
            if path.relative_to(wiki).parts[0] not in CATEGORIES:
                fail(path, 'knowledge page outside the five categories')
            if path.stem in pages:
                fail(path, 'duplicate filename: ' + path.stem)
            pages[path.stem] = path
            counts['knowledge_pages'] += 1
            if 'slug' in meta:
                fail(path, 'redundant slug field')
            if re.search(r'(?:^|[\s/`(\"\'])raw/', body):
                fail(path, 'raw path in knowledge body')
            if '## 来源身份' not in body:
                fail(path, 'missing source identity section')
        if 'updated' in meta:
            try:
                if date.fromisoformat(meta['updated']) > date.today():
                    fail(path, 'future updated date')
            except (ValueError, TypeError):
                fail(path, 'invalid updated date')
        for key in ('tags', 'aliases'):
            if key in meta and not isinstance(meta[key], list):
                fail(path, key + ' must be a string list')
        tags = meta.get('tags', [])
        if isinstance(tags, list):
            if not is_index and not tags:
                fail(path, 'knowledge page needs nonempty tags')
            if len(tags) != len(set(tags)):
                fail(path, 'duplicate tag')
        for tag in meta.get('tags', []) if isinstance(meta.get('tags', []), list) else []:
            if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', tag):
                fail(path, 'invalid tag: ' + tag)
        sources = meta.get('sources', [])
        if not isinstance(sources, list):
            fail(path, 'sources must be a string list')
            sources = []
        if not sources and meta.get('type') != 'index':
            fail(path, 'knowledge page needs direct sources')
        if len(sources) != len(set(sources)):
            fail(path, 'duplicate source registration')
        known_ids = set()
        for source in sources:
            counts['source_entries'] += 1
            if PINNED_CODE.fullmatch(source):
                counts['pinned_external_sources'] += 1
                continue
            if not source.startswith(('raw/', 'research/')) or any(c in source for c in ['*', '?', '\\']):
                fail(path, 'invalid source registration: ' + source)
                continue
            target = (root / source).resolve()
            base = root / source.split('/')[0]
            if not target.is_relative_to(base):
                fail(path, 'source path escapes its root: ' + source)
                continue
            if source.startswith('raw/') and not raw_present:
                counts['local_sources_skipped'] += 1
                continue
            if not target.is_file():
                fail(path, 'source file missing: ' + source)
                continue
            counts['local_sources_checked'] += 1
            if source not in source_cache:
                found = set()
                for parent in target.parents:
                    if parent == root:
                        break
                    record = parent / 'README.md'
                    if record.is_file():
                        mt = record.read_text()
                        if mt.startswith('---\n'):
                            found = set(ARXIV.findall(mt.split('---', 2)[1]))
                            break
                source_cache[source] = found
            known_ids |= source_cache[source]
        if raw_present:
            for aid in set(ARXIV.findall(body)) - known_ids:
                fail(path, 'arXiv ID not present in registered source metadata: ' + aid)
        counts['numbered_citations'] += len(re.findall(r'(附录\s?[A-Z]|§\s?\d+(?:\.\d+)*|表\s?\d+|图\s?\d+)', body))

    # Project entrances and installed project skill documentation share moved links.
    extra = [root / n for n in ['README.md', 'AGENTS.md', 'CLAUDE.md']]
    extra += list((root / 'research').rglob('*.md'))
    extra += list((root / '.agents/skills').rglob('*.md'))
    for path in extra:
        if path.is_file():
            bodies[path] = path.read_text()
    for path, body in bodies.items():
        visible = prose(body)
        graph[path] = set()
        if re.search(r'\[\[[^\]\n]+\]\]', visible):
            fail(path, 'legacy wikilink remains')
        for dest in re.findall(r'!?\[[^\]\n]*\]\(([^\n)]+)\)', visible):
            dest = dest.strip('<>')
            if re.match(r'^[a-zA-Z][\w+.-]*:', dest):
                continue
            file, _, fragment = unquote(dest).partition('#')
            target = (path.parent / file).resolve() if file else path
            if not target.exists():
                fail(path, 'missing Markdown target: ' + dest)
                continue
            if target.suffix == '.md':
                graph[path].add(target)
                if fragment and fragment not in anchors(target.read_text()):
                    fail(path, 'missing heading: ' + dest)
            counts['local_links'] += 1
        if path.is_relative_to(wiki):
            width = None
            for line in visible.splitlines():
                if line.startswith('|'):
                    current = len(re.split(r'(?<!\\)\|', line)[1:-1])
                    if width is not None and width != current:
                        fail(path, 'table width mismatch')
                    width = current
                else:
                    width = None
    reached, pending = set(), [wiki / 'INDEX.md']
    while pending:
        p = pending.pop()
        if p in reached:
            continue
        reached.add(p)
        pending.extend(graph.get(p, set()) - reached)
    for slug, path in pages.items():
        if path not in reached:
            fail(path, 'not reachable from INDEX')
        incoming = any(path in graph.get(q, set()) for q in pages.values() if q != path)
        outgoing = bool((graph.get(path, set()) - {path}) & set(pages.values()))
        if not incoming or not outgoing:
            warnings.append(slug + ': review missing knowledge incoming/outgoing links')
    return dict(status='failed' if errors else 'passed', counts=dict(counts), errors=errors,
                warnings=warnings, skipped=skipped,
                scope='Mechanical structure, local links/anchors, source metadata and navigation; no scientific or runtime verification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Project directory containing wiki/ and optional raw/')
    parser.add_argument('--require-raw', action='store_true')
    args = parser.parse_args()
    result = check(args.root, args.require_raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
