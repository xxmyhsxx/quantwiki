"""Read-only Wiki structure, Markdown, source identity and navigation checks.

Uses the documented scalar and string-list frontmatter subset; no third-party
packages. --links-only checks Wiki click boundaries and targets, optionally
limited by --paths. --raw-links-only checks raw storage and its no-link rule.
--strict also fails on items needing review. No scientific,
relationship-purpose or client-jump verification is performed.
"""
import argparse
from collections import Counter
from datetime import date
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

CATEGORIES = {'fundamentals', 'theory', 'methods', 'research', 'implementation'}
TYPES = {'source', 'concept', 'method', 'implementation', 'comparison', 'recipe', 'question', 'index'}
PINNED_CODE = re.compile(r'https://github\.com/[^/]+/[^/]+/(?:blob|tree)/[0-9a-f]{40}(?:/[^\s]+)?$')
ARXIV = re.compile(r'(?:arXiv:\s*|arxiv\.org/(?:abs|pdf|src)/)(\d{4}\.\d{4,5})')


BACKTICK = chr(96)
LINK_LIMITS = [
    'Excluded: frontmatter, fenced/indented code, inline code, comments and unused reference definitions.',
    'Supported: inline/reference Markdown links and images, autolinks, common bare URLs, HTML a[href]/img[src].',
    'Other Markdown extensions, HTML attributes and every malformed construct are not covered.',
    'Heading checks are static; Obsidian/client jumps and relationship purpose require contextual review.',
]


def mask(text):
    return ''.join('\n' if char == '\n' else ' ' for char in text)


def body_text(text):
    """Mask examples and frontmatter while retaining original line positions."""
    lines = text.splitlines(keepends=True)
    front = bool(lines and lines[0].strip() == '---')
    fence, out = None, []
    in_list, previous_blank, indented_code = False, True, False
    for index, line in enumerate(lines):
        if front:
            out.append(mask(line))
            if index and line.strip() in {'---', '...'}:
                front = False
            continue
        content = re.sub(r'^(?: {0,3}>[ \t]?)+', '', line)
        list_start = re.match(r'^ {0,3}(?:[-+*]|\d+[.)])\s+(.*)$', content)
        if list_start:
            in_list = True
            content = list_start[1]
        indent = len(content) - len(content.lstrip(' \t'))
        opening = re.match(r'^[ \t]*(' + BACKTICK + r'{3,}|~{3,})(.*)$', content)
        if indent > 3 and not in_list:
            opening = None
        if fence:
            out.append(mask(line))
            if re.match(r'^[ \t]*' + re.escape(fence[0]) +
                        '{' + str(fence[1]) + r',}\s*$', content):
                fence = None
        elif opening:
            fence = (opening[1][0], len(opening[1]))
            out.append(mask(line))
        elif line.startswith(('    ', '\t')) and (indented_code or (previous_blank and not in_list)):
            out.append(mask(line))
            indented_code = True
        else:
            out.append(line)
            if line.strip() and not line.startswith(('    ', '\t')):
                in_list = bool(re.match(r'^ {0,3}(?:[-+*]|\d+[.)])\s+', line))
                indented_code = False
        previous_blank = not line.strip()
    return re.sub(r'<!--[\s\S]*?(?:-->|$)', lambda m: mask(m[0]), ''.join(out))


def prose(text):
    text = body_text(text)
    pattern = re.compile(r'(?<![\\' + BACKTICK + '])(' + BACKTICK + r'+)(?!' + BACKTICK + ')')
    cursor, parts = 0, []
    while True:
        start = pattern.search(text, cursor)
        if not start:
            parts.append(text[cursor:])
            break
        end = re.search(r'(?<!' + BACKTICK + ')' + re.escape(start[1]) +
                        '(?!' + BACKTICK + ')', text[start.end():])
        if not end:
            parts.append(text[cursor:start.end()])
            cursor = start.end()
            continue
        stop = start.end() + end.end()
        parts.extend((text[cursor:start.start()], mask(text[start.start():stop])))
        cursor = stop
    return ''.join(parts)


def heading_info(text):
    counts = Counter()
    explicit = set(re.findall(r'<a\s+[^>]*?(?:id|name)=["\']([^"\']+)', prose(text)))
    slugs, titles = {}, set()
    for h in re.findall(r'^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$', body_text(text), re.M):
        h = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', h)
        h = re.sub(BACKTICK + r'+([^' + BACKTICK + ']+)' + BACKTICK + '+', r'\1', h)
        h = html.unescape(h)
        titles.add(h)
        base = re.sub(r'[^\w\-\s]', '', h.lower()).replace(' ', '-')
        slugs[base + ('-' + str(counts[base]) if counts[base] else '')] = h
        counts[base] += 1
    return slugs, titles, explicit


def anchors(text):
    slugs, titles, explicit = heading_info(text)
    return set(slugs) | titles | explicit


def unescape(value):
    return html.unescape(re.sub(r'\\([!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_' +
                               BACKTICK + r'{|}~])', r'\1', value))


def destination(text, start):
    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor == len(text):
        return None
    first = cursor
    if text[cursor] == '<':
        cursor += 1
        first = cursor
        while cursor < len(text):
            if text[cursor] == '\\':
                cursor += 2
                continue
            if text[cursor] == '>':
                return unescape(text[first:cursor]), cursor + 1
            if text[cursor] == '\n':
                return None
            cursor += 1
        return None
    depth = 0
    while cursor < len(text):
        char = text[cursor]
        if char == '\\' and cursor + 1 < len(text):
            cursor += 2
            continue
        if char.isspace() or (char == ')' and depth == 0):
            break
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        cursor += 1
    return None if depth else (unescape(text[first:cursor]), cursor)


def closing_link(text, cursor):
    before = cursor
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor < len(text) and text[cursor] == ')':
        return cursor + 1
    if cursor == before or cursor == len(text) or text[cursor] not in "\"'(":
        return None
    closer = ')' if text[cursor] == '(' else text[cursor]
    cursor += 1
    while cursor < len(text):
        if text[cursor] == '\\':
            cursor += 2
        elif text[cursor] == closer:
            cursor += 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            return cursor + 1 if cursor < len(text) and text[cursor] == ')' else None
        else:
            cursor += 1
    return None


def bracket(text, start):
    depth, cursor = 1, start + 1
    while cursor < len(text):
        if text[cursor] == '\\':
            cursor += 2
            continue
        if text[cursor] == '[':
            depth += 1
        elif text[cursor] == ']':
            depth -= 1
            if depth == 0:
                return text[start + 1:cursor], cursor + 1
        cursor += 1
    return None


def reference_id(label):
    return ' '.join(unescape(label).split()).casefold()


@dataclass(frozen=True)
class Link:
    target: str
    line: int
    kind: str
    image: bool = False


def links(text, reference_definitions=None, invalid_references=None):
    """Parse the declared subset; return recognized but unparsed candidates."""
    text = prose(text)
    found, candidates = [], []

    def line_at(offset):
        return text.count('\n', 0, offset) + 1

    for m in re.finditer(r'\[\[[^\]\n]+\]\]', text):
        candidates.append(Link(m[0], line_at(m.start()), 'wikilink'))
    text = re.sub(r'\[\[[^\]\n]+\]\]', lambda m: mask(m[0]), text)
    definitions = dict(reference_definitions or {})
    invalid_definitions = set(invalid_references or ())
    definition_pattern = r'^ {0,3}\[([^\]\n]+)\]:[ \t]*(.*)$'
    for m in re.finditer(definition_pattern, text, re.M):
        if not m[1].startswith('^'):
            parsed = destination(m[2], 0)
            if parsed:
                definitions.setdefault(reference_id(m[1]), parsed[0])
            else:
                invalid_definitions.add(reference_id(m[1]))
    text = re.sub(definition_pattern,
                  lambda m: m[0] if m[1].startswith('^') else mask(m[0]), text, flags=re.M)
    spans, cursor = [], 0
    while cursor < len(text):
        if text[cursor] == '\\':
            cursor += 2
            continue
        if text[cursor] != '[':
            cursor += 1
            continue
        label = bracket(text, cursor)
        if not label:
            cursor += 1
            continue
        value, end = label
        target, stop, kind = None, end, 'reference'
        if end < len(text) and text[end] == '(':
            kind = 'inline'
            parsed = destination(text, end + 1)
            close = closing_link(text, parsed[1]) if parsed else None
            if close:
                target, stop = parsed[0], close
            else:
                candidates.append(Link(text[cursor:end + 1], line_at(cursor), 'unparsed'))
        elif end < len(text) and text[end] == '[':
            ref = bracket(text, end)
            if ref:
                target = definitions.get(reference_id(ref[0] or value))
                stop = ref[1]
                if target is None and reference_id(ref[0] or value) in invalid_definitions:
                    candidates.append(Link(text[cursor:stop], line_at(cursor), 'unparsed'))
        else:
            target = definitions.get(reference_id(value))
            if target is None and reference_id(value) in invalid_definitions:
                candidates.append(Link(text[cursor:stop], line_at(cursor), 'unparsed'))
        if target is not None:
            image = cursor > 0 and text[cursor - 1] == '!'
            found.append(Link(target, line_at(cursor), kind, image))
            if '![' in value:
                nested, unparsed = links(value, definitions, invalid_definitions)
                found.extend(Link(item.target, line_at(cursor) + item.line - 1, item.kind, True)
                             for item in nested if item.image)
                candidates.extend(Link(item.target, line_at(cursor) + item.line - 1, item.kind)
                                  for item in unparsed)
            spans.append((cursor, stop))
        cursor = stop
    for start, stop in reversed(spans):
        text = text[:start] + mask(text[start:stop]) + text[stop:]
    for m in re.finditer(r'<(a|img)\b([^>]*)>', text, re.I):
        attr = 'href' if m[1].lower() == 'a' else 'src'
        value = re.search(r'\b' + attr + r'\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
                          m[2], re.I)
        if value:
            target = next(v for v in value.groups() if v is not None)
            found.append(Link(html.unescape(target), line_at(m.start()), 'html', attr == 'src'))
        if re.search(r'\bsrcset\s*=', m[2], re.I):
            candidates.append(Link(m[0], line_at(m.start()), 'unparsed'))
    text = re.sub(r'<a\b[^>]*>[\s\S]*?</a>', lambda m: mask(m[0]), text, flags=re.I)
    text = re.sub(r'<(?:a|img)\b[^>]*>', lambda m: mask(m[0]), text, flags=re.I)
    for m in re.finditer(r'<([A-Za-z][\w+.-]*:[^<>\s]*|[^<>\s@]+@[^<>\s@]+\.[^<>\s@]+)>', text):
        target = m[1] if ':' in m[1] else 'mailto:' + m[1]
        found.append(Link(target, line_at(m.start()), 'autolink'))
    text = re.sub(r'<([A-Za-z][\w+.-]*:[^<>\s]*|[^<>\s@]+@[^<>\s@]+\.[^<>\s@]+)>',
                  lambda m: mask(m[0]), text)
    for m in re.finditer(r'(?<![\w])(?:(?:https?|ftp)://|www\.)[^\s<>"' +
                         BACKTICK + r']+', text, re.I):
        target = m[0].rstrip('.,;:!?，。；：！？、')
        while target and target[-1] in ')]}':
            closer = target[-1]
            if target.count(closer) <= target.count({')': '(', ']': '[', '}': '{'}[closer]):
                break
            target = target[:-1]
        found.append(Link(target, line_at(m.start()), 'bare_url'))
    return found, candidates


def inspect_links(root, bodies):
    wiki = root / 'wiki'
    graph, counts, errors, warnings, findings, cache = {}, Counter(), [], [], [], {}

    def issue(path, link, level, code, message):
        relative = str(path.relative_to(root))
        finding = dict(path=relative, line=link.line, target=link.target,
                       syntax=link.kind, level=level, code=code, message=message)
        findings.append(finding)
        (errors if level == 'error' else warnings).append(
            relative + ':' + str(link.line) + ': ' + message + ': ' + link.target)

    for path, text in bodies.items():
        graph[path] = set()
        found, candidates = links(text)
        counts['markdown_files'] += 1
        for link in candidates:
            if link.kind == 'wikilink':
                issue(path, link, 'error', 'wikilink', 'use a standard Markdown relative link')
            else:
                issue(path, link, 'warning', 'unparsed_link', 'recognized link candidate requires direct review')
        for link in found:
            counts['clickable_references'] += 1
            counts[link.kind + '_references'] += 1
            in_wiki = path.is_relative_to(wiki)
            if re.match(r'^[a-zA-Z][\w+.-]*:', link.target) or link.kind == 'bare_url':
                counts['external_references'] += 1
                if in_wiki:
                    issue(path, link, 'error', 'outside_wiki', 'Wiki body click target must remain inside wiki/')
                else:
                    counts['external_references_skipped'] += 1
                continue
            file, _, fragment = link.target.partition('#')
            file, fragment = unquote(file), unquote(fragment)
            target = (path.parent / file).resolve() if file else path.resolve()
            counts['local_links'] += 1
            if in_wiki and not target.is_relative_to(wiki):
                issue(path, link, 'error', 'outside_wiki', 'resolved target leaves wiki/')
                continue
            if in_wiki and Path(file).is_absolute():
                issue(path, link, 'error', 'absolute_path', 'Wiki links must use relative paths')
                continue
            if not target.exists() or (in_wiki and not target.is_file()):
                issue(path, link, 'error', 'missing_target', 'target does not exist or is not an allowed file')
                continue
            if not target.is_file():
                # Project documentation may link to an existing directory.
                continue
            if target.suffix == '.md' and not link.image:
                graph[path].add(target)
                if fragment:
                    if fragment.startswith('^'):
                        issue(path, link, 'warning', 'unsupported_anchor', 'block-reference anchors are not statically checked')
                        continue
                    if target not in cache:
                        cache[target] = heading_info(target.read_text())
                    slugs, titles, explicit = cache[target]
                    if fragment not in set(slugs) | titles | explicit:
                        issue(path, link, 'error', 'missing_heading', 'heading or explicit anchor does not exist')
                    elif in_wiki and fragment in slugs and fragment not in titles | explicit:
                        issue(path, link, 'warning', 'heading_compatibility',
                              'slug differs from heading text; use the encoded heading or verify the client jump')
    return graph, counts, errors, warnings, findings


def check_link_scope(root, paths=None):
    root = Path(root).resolve()
    wiki = root / 'wiki'
    selected, selection_errors = set(), []
    if not wiki.is_dir():
        selection_errors.append('missing wiki/ directory')
    elif paths is None:
        selected.update(p for p in wiki.rglob('*.md') if p.is_file())
    else:
        for value in paths:
            target = (root / value).resolve()
            if not target.is_relative_to(wiki):
                selection_errors.append('selected path must remain inside wiki/: ' + value)
            elif not target.exists():
                selection_errors.append('selected path does not exist: ' + value)
            else:
                found = {p for p in target.rglob('*.md') if p.is_file()} if target.is_dir() else (
                    {target} if target.suffix == '.md' and target.is_file() else set())
                if not found:
                    selection_errors.append('selected path contains no Markdown files: ' + value)
                selected.update(found)
    if not selected and not selection_errors:
        selection_errors.append('no Wiki Markdown files selected')
    bodies = {}
    for path in sorted(selected):
        if not path.resolve().is_relative_to(wiki):
            selection_errors.append('selected Markdown symlink leaves wiki/: ' + str(path.relative_to(root)))
        else:
            bodies[path] = path.read_text()
    _, counts, errors, warnings, findings = inspect_links(root, bodies)
    errors = selection_errors + errors
    return dict(status='failed' if errors else 'needs_review' if warnings else 'passed',
                counts=dict(counts), errors=errors, warnings=warnings, findings=findings,
                files=[str(p.relative_to(root)) for p in sorted(bodies)],
                scope='Wiki body clickable references only; no raw/source completeness or scientific review.',
                skipped=LINK_LIMITS, semantic_review='Not performed: read source context and relevant target content.')


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


def raw_record(path, raw):
    """Only local management and source-entry records remain Markdown."""
    parts = path.relative_to(raw).parts
    return (len(parts) == 1 and parts[0] in {'README.md', 'LOG.md'}) or (
        len(parts) == 4 and parts[0] in {'papers', 'articles', 'repositories'}
        and parts[-1] == 'README.md')


def check_raw_links(root):
    root = Path(root).resolve()
    raw = root / 'raw'
    counts, errors, warnings, findings, files = Counter(), [], [], [], []

    def issue(path, code, message, link=None, level='error'):
        relative = str(path.relative_to(root))
        finding = dict(path=relative, line=link.line if link else 1,
                       target=link.target if link else '',
                       syntax=link.kind if link else 'storage',
                       level=level, code=code, message=message)
        findings.append(finding)
        (errors if level == 'error' else warnings).append(
            relative + ':' + str(finding['line']) + ': ' + message)

    if not raw.is_dir():
        issue(raw, 'raw_missing', 'raw is absent; no-link storage was not checked')
    elif not raw.resolve().is_relative_to(root):
        issue(raw, 'raw_outside_root', 'raw resolves outside the project')
    else:
        def walk_error(error):
            issue(Path(error.filename), 'raw_unreadable', str(error))

        for directory, dirs, names in os.walk(raw, followlinks=False, onerror=walk_error):
            directory_parts = Path(directory).relative_to(raw).parts
            entry_root = (len(directory_parts) == 3 and
                          directory_parts[0] in {'papers', 'articles'})
            if (len(directory_parts) == 4 and directory_parts[0] in {'papers', 'articles'}
                    and directory_parts[-1] == 'assets' and 'pdf-pages' in dirs
                    and any((Path(directory).parent / p).is_file()
                            for p in ('paper.pdf', 'lecture.pdf'))):
                counts['raw_generated_copies'] += 1
                issue(Path(directory) / 'pdf-pages', 'raw_generated_copy',
                      'PDF page renders duplicate the original; retain the source PDF')
            # Git objects are original provenance, not expanded reading pages.
            for name in dirs:
                path = Path(directory) / name
                if name != '.git' and path.is_symlink() and (
                        not path.exists() or not path.resolve().is_relative_to(raw.resolve())):
                    issue(path, 'raw_directory_symlink',
                          'directory symlink is missing or resolves outside the checked raw tree')
            dirs[:] = sorted(d for d in dirs if d != '.git' and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                path = Path(directory) / name
                generated = {'formulas.md', 'formulas.md.txt', 'figures.md', 'figures.md.txt',
                             'formula-context.tex'}
                if entry_root and (name in generated or (
                        directory_parts[0] == 'papers' and name in {
                            'paper.md', 'paper.md.txt', 'paper-readable.pdf'}
                        and (Path(directory) / 'paper.pdf').is_file()) or (
                        directory_parts[0] == 'articles' and name in {'article.md', 'article.md.txt'}
                        and (Path(directory) / 'lecture.pdf').is_file())) or (
                        len(directory_parts) == 4 and directory_parts[0] == 'articles'
                        and directory_parts[-1] == 'source'
                        and name in {'article-math-reference.md', 'article-math-reference.md.txt'}
                        and (Path(directory).parent / 'article.md.txt').is_file()):
                    counts['raw_generated_copies'] += 1
                    issue(path, 'raw_generated_copy',
                          'generated reader duplicates source material; keep explanations in Wiki/research')
                if name.lower().endswith(('.md.symlink.txt', '.markdown.symlink.txt')):
                    counts['raw_sealed_symlink_files'] += 1
                    continue
                if name.lower().endswith(('.md.txt', '.markdown.txt')):
                    counts['raw_sealed_text_files'] += 1
                    continue
                if path.suffix.lower() not in {'.md', '.markdown'}:
                    continue
                files.append(str(path.relative_to(root)))
                counts['raw_markdown_files'] += 1
                if not raw_record(path, raw):
                    counts['raw_unsealed_markdown_files'] += 1
                    issue(path, 'raw_unsealed_markdown',
                          'source or reading Markdown must be stored as original-name + .txt')
                    continue
                if not path.resolve().is_relative_to(raw):
                    issue(path, 'raw_outside_root', 'raw record resolves outside raw')
                    continue
                try:
                    found, candidates = links(path.read_text())
                except (OSError, UnicodeError) as error:
                    issue(path, 'raw_unreadable', str(error))
                    continue
                counts['raw_clickable_references'] += len(found)
                for link in found:
                    issue(path, 'raw_clickable_reference',
                          'raw records must use code text for URLs and file locations', link)
                for link in candidates:
                    issue(path, 'raw_clickable_reference' if link.kind == 'wikilink' else 'unparsed_link',
                          'raw link candidate needs removal or review', link,
                          'error' if link.kind == 'wikilink' else 'warning')
    return dict(status='failed' if errors else 'needs_review' if warnings else 'passed',
                counts=dict(counts), errors=errors, warnings=warnings, findings=findings,
                files=files, skipped=LINK_LIMITS + [
                    'Sealed .md.txt/.markdown.txt bytes and original Git object directories are not read.',
                    'File hashes, original source completeness and application graph state are not verified.',
                ], scope='raw storage extensions, known generated copies and no clicks in retained Markdown records.',
                semantic_review='Not performed: no source fidelity or scientific review.')


def check(root, require_raw=False):
    root = Path(root).resolve()
    wiki = root / 'wiki'
    errors, warnings, skipped = [], [], []
    counts = Counter()
    pages, bodies, graph, source_cache, paper_cache = {}, {}, {}, {}, {}
    raw_present = (root / 'raw').is_dir()
    if not raw_present:
        (errors if require_raw else skipped).append('raw absent: local source files and raw metadata not verified')

    def fail(path, message):
        errors.append(str(path.relative_to(root)) + ': ' + message)

    for name in ('README.md', 'INDEX.md', 'READING.md', 'LOG.md'):
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
        bodies[path] = text
        is_index = path.parent == wiki and path.name in {'INDEX.md', 'READING.md'}
        for key in ('title', 'type', 'sources') + (() if is_index else ('tags',)):
            if key not in meta:
                fail(path, 'missing field: ' + key)
        if not isinstance(meta.get('title'), str) or not meta.get('title', '').strip():
            fail(path, 'title must be a nonempty string')
        if meta.get('type') not in TYPES:
            fail(path, 'invalid page type')
        if is_index and meta.get('type') != 'index':
            fail(path, path.name + ' must have type: index')
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
        known_ids, registered_papers = set(), {}
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
                paper_cache[source] = None
                for parent in target.parents:
                    if parent == root:
                        break
                    record = parent / 'README.md'
                    if record.is_file():
                        mt = record.read_text()
                        if mt.startswith('---\n'):
                            found = set(ARXIV.findall(mt.split('---', 2)[1]))
                            record_meta, _, _ = metadata(mt)
                            if record_meta.get('kind') == 'paper' and record_meta.get('source_id'):
                                paper_cache[source] = ('paper', record_meta['source_id'])
                            break
                source_cache[source] = found
            known_ids |= source_cache[source]
            identity = paper_cache[source]
            if identity:
                if identity in registered_papers:
                    counts['duplicate_paper_carriers'] += 1
                    fail(path, 'duplicate carriers for paper source ' + identity[1] + ': ' +
                         registered_papers[identity] + ' and ' + source)
                else:
                    registered_papers[identity] = source
        if raw_present and not is_index:
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
    graph, link_counts, link_errors, link_warnings, findings = inspect_links(root, bodies)
    counts.update(link_counts)
    errors.extend(link_errors)
    warnings.extend(link_warnings)
    for path, body in bodies.items():
        visible = prose(body)
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
    raw_files, raw_limits = [], []
    if raw_present:
        raw_result = check_raw_links(root)
        counts.update(raw_result['counts'])
        errors.extend(raw_result['errors'])
        warnings.extend(raw_result['warnings'])
        findings.extend(raw_result['findings'])
        raw_files = raw_result['files']
        raw_limits = raw_result['skipped']
    return dict(status='failed' if errors else 'needs_review' if warnings else 'passed',
                counts=dict(counts), errors=errors, warnings=warnings, findings=findings,
                files=[str(p.relative_to(root)) for p in sorted(bodies)] + raw_files,
                skipped=list(dict.fromkeys(skipped + LINK_LIMITS + raw_limits)),
                scope='Mechanical structure, Wiki click boundary, raw no-link storage, source metadata and navigation.',
                semantic_review='Not performed: no scientific, relationship-purpose or runtime verification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Project directory containing wiki/ and optional raw/')
    parser.add_argument('--require-raw', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--links-only', action='store_true',
                      help='Check only Wiki body click boundaries and targets; no source or structure checks')
    mode.add_argument('--raw-links-only', action='store_true',
                      help='Check raw no-link records and sealed Markdown storage; no source-content checks')
    parser.add_argument('--paths', nargs='+',
                        help='Wiki files/directories relative to --root; requires --links-only')
    parser.add_argument('--strict', action='store_true',
                        help='Exit 1 on warnings/needs-review items as well as errors')
    args = parser.parse_args()
    if args.paths and not args.links_only:
        parser.error('--paths requires --links-only')
    if (args.links_only or args.raw_links_only) and args.require_raw:
        parser.error('--require-raw cannot be combined with a links-only mode')
    result = (check_raw_links(args.root) if args.raw_links_only else
              check_link_scope(args.root, args.paths) if args.links_only else check(args.root, args.require_raw))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result['errors'] or (args.strict and result['warnings']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
