"""Regression tests for Wiki click lint and its full-check integration."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check_wiki import check, check_link_scope

SCRIPT = Path(__file__).with_name('check_wiki.py')
BT = chr(96)


class LinkLintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.write('wiki/methods/page.md', '# Page\n')
        self.write('wiki/methods/target.md',
                   '# Target\n## A heading\n## 7. 剪枝额外改变序列集合、深度与位置\n'
                   '### 7.1 attention 总量、集中度与所关注对象是三件事\n'
                   '<a id="fixed"></a>\n')
        self.write('wiki/INDEX.md', '# Index\n')
        self.write('wiki/assets/image.png', 'image')
        self.write('raw/source.md', '# Source\n')
        self.write('README.md', '# Project\n')

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def lint(self, text):
        self.write('wiki/methods/page.md', text)
        return check_link_scope(self.root, ['wiki/methods/page.md'])

    def codes(self, result, level='error'):
        return [f['code'] for f in result['findings'] if f['level'] == level]

    def cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(SCRIPT), '--root', str(self.root), *args],
                              capture_output=True, text=True)

    def test_valid_internal_navigation_assets_and_headings(self):
        result = self.lint('# Page\n[nav](../INDEX.md)\n[section](target.md#a-heading)\n'
                           '[same](#Page)\n[explicit](target.md#fixed)\n'
                           '![image](../assets/image.png)\n<img src="../assets/image.png">\n'
                           '<a href="target.md">target</a>\n')
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['counts']['clickable_references'], 7)

    def test_encoded_chinese_spaces_parentheses_and_hash_filename(self):
        self.write('wiki/methods/中文 名称.md', '# 中文标题\n')
        self.write('wiki/methods/name(test).md', '# Test\n')
        self.write('wiki/methods/hash#file.md', '# Hash\n')
        result = self.lint('[a](<中文 名称.md#中文标题>)\n'
                           '[b](%E4%B8%AD%E6%96%87%20%E5%90%8D%E7%A7%B0.md#%E4%B8%AD%E6%96%87%E6%A0%87%E9%A2%98)\n'
                           '[c](name(test).md "Title")\n[d](hash%23file.md)\n')
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['counts']['local_links'], 4)

    def test_external_protocol_autolink_bare_and_html_are_rejected(self):
        samples = ['[x](https://example.org)', '[x](mailto:x@example.org)',
                   '<https://example.org>', '<x@example.org>', 'https://example.org。',
                   'www.example.org', '<a href="https://example.org">source</a>',
                   '<img src="https://example.org/img.png">', '![x](https://example.org/img.png)']
        for sample in samples:
            with self.subTest(sample=sample):
                result = self.lint(sample)
                self.assertEqual(self.codes(result), ['outside_wiki'], result)

    def test_bare_url_punctuation_preserves_balanced_parentheses(self):
        result = self.lint('See (https://example.org/a(b)).\n')
        self.assertEqual(result['findings'][0]['target'], 'https://example.org/a(b)')

    def test_raw_project_root_absolute_and_symlink_targets_are_rejected(self):
        outside = self.root / 'raw/source.md'
        (self.root / 'wiki/methods/escape.md').symlink_to(outside)
        for target in ['../../raw/source.md', '../../README.md', str(outside), 'escape.md']:
            with self.subTest(target=target):
                self.assertEqual(self.codes(self.lint('[x](' + target + ')')), ['outside_wiki'])

    def test_source_metadata_code_comments_and_unused_definitions_are_excluded(self):
        result = self.lint('---\nsources:\n  - raw/missing.pdf\n'
                           '  - https://example.org/source\n---\n'
                           + BT + 'https://example.org/identity' + BT + '\n'
                           + BT * 4 + 'markdown\n[x](https://example.org)\n' + BT * 4 + '\n'
                           '<!-- [x](https://example.org) -->\n'
                           '\n    [example](https://example.org)\n'
                           '[unused]: https://example.org/not-clicked\n'
                           '[real](target.md)\n')
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['counts']['clickable_references'], 1)

    def test_indented_list_continuation_is_checked(self):
        result = self.lint('1. Explanation\n\n    [source](https://example.org)\n')
        self.assertEqual(self.codes(result), ['outside_wiki'], result)

    def test_list_and_blockquote_fenced_examples_are_excluded(self):
        for sample in ['1. Example\n\n    ~~~markdown\n    [source](https://example.org)\n    ~~~\n',
                       '1. ~~~markdown\n    [source](https://example.org)\n    ~~~\n',
                       '> ~~~markdown\n> [source](https://example.org)\n> ~~~\n']:
            with self.subTest(sample=sample):
                result = self.lint(sample + '[real](target.md)\n')
                self.assertEqual(result['status'], 'passed', result)
                self.assertEqual(result['counts']['clickable_references'], 1)

    def test_reference_use_is_counted_not_definition(self):
        result = self.lint('[one][ref]\n[ref][]\n[ref]\n![image][pic]\n'
                           '[ref]: target.md "Title"\n[pic]: ../assets/image.png\n'
                           '[unused]: https://example.org\n')
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['counts']['reference_references'], 4)

    def test_external_reference_and_footnote_body_are_checked(self):
        result = self.lint('[source][s]\n[s]: https://example.org\n'
                           '[^1]: [footnote source](https://example.org/note)\n')
        self.assertEqual(self.codes(result), ['outside_wiki', 'outside_wiki'], result)

    def test_plain_citation_markers_and_math_indices_are_not_links(self):
        result = self.lint('Claim.[S1][S3][VEP26，decoder 构造]\n$a[0][1]$\n')
        self.assertEqual(result['status'], 'passed', result)
        self.assertNotIn('clickable_references', result['counts'])

    def test_used_unparsed_definition_is_disclosed(self):
        result = self.lint('[source][ref]\n[ref]: <unfinished\n')
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertEqual(self.codes(result, 'warning'), ['unparsed_link'])

    def test_linked_image_checks_both_targets(self):
        result = self.lint('[![image](../assets/image.png)](target.md)\n')
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['counts']['clickable_references'], 2)
        self.assertEqual(self.codes(self.lint('[![image](../../raw/source.md)](target.md)')),
                         ['outside_wiki'])

    def test_linked_reference_image_inherits_outer_definitions(self):
        result = self.lint('[![image][pic]](target.md)\n[pic]: https://example.org/image.png\n')
        self.assertEqual(self.codes(result), ['outside_wiki'], result)
        self.assertEqual(result['counts']['clickable_references'], 2)
        result = self.lint('[![image][pic]](target.md)\n[pic]: ../assets/image.png\n')
        self.assertEqual(result['status'], 'passed', result)

    def test_missing_file_heading_and_wikilink_are_distinct_errors(self):
        result = self.lint('[missing](missing.md)\n[heading](target.md#absent)\n[[target]]\n')
        self.assertCountEqual(self.codes(result), ['missing_target', 'missing_heading', 'wikilink'])
        by_code = {f['code']: f['line'] for f in result['findings']}
        self.assertEqual(by_code, {'missing_target': 1, 'missing_heading': 2, 'wikilink': 3})

    def test_empty_missing_and_outside_selections_cannot_pass(self):
        (self.root / 'wiki/empty').mkdir()
        for selection in [[], ['wiki/empty'], ['wiki/missing.md'], ['README.md'], ['wiki/assets/image.png']]:
            with self.subTest(selection=selection):
                self.assertEqual(check_link_scope(self.root, selection)['status'], 'failed')

    def test_default_scope_does_not_read_symlinked_document_outside_wiki(self):
        (self.root / 'wiki/escape.md').symlink_to(self.root / 'raw/source.md')
        result = check_link_scope(self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('wiki/escape.md', result['files'])

    def test_saprune_slug_compatibility_is_needs_review_not_confirmed_error(self):
        result = self.lint('[a](target.md#7-剪枝额外改变序列集合深度与位置)\n'
                           '[b](target.md#71-attention-总量集中度与所关注对象是三件事)\n')
        self.assertFalse(result['errors'], result)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(self.codes(result, 'warning'), ['heading_compatibility'] * 2)
        self.assertEqual(self.cli('--links-only', '--paths', 'wiki/methods/page.md').returncode, 0)
        strict = self.cli('--links-only', '--paths', 'wiki/methods/page.md', '--strict')
        self.assertEqual(strict.returncode, 1)
        self.assertEqual(json.loads(strict.stdout)['status'], 'needs_review')

    def test_exact_chinese_heading_has_no_slug_compatibility_warning(self):
        result = self.lint('[a](<target.md#7. 剪枝额外改变序列集合、深度与位置>)\n')
        self.assertEqual(result['status'], 'passed', result)

    def test_identifiable_unparsed_candidate_and_block_anchor_are_disclosed(self):
        result = self.lint('[bad](<unfinished\n[unknown][absent]\n[block](target.md#^block)\n')
        self.assertFalse(result['errors'], result)
        self.assertIn('unparsed_link', self.codes(result, 'warning'))
        self.assertIn('unsupported_anchor', self.codes(result, 'warning'))
        self.assertEqual(result['status'], 'needs_review')

    def test_invalid_cli_combinations_exit_two(self):
        self.assertEqual(self.cli('--paths', 'wiki/INDEX.md').returncode, 2)
        self.assertEqual(self.cli('--links-only', '--require-raw').returncode, 2)

    def test_full_check_preserves_sources_navigation_and_nonwiki_local_checks(self):
        self.write('raw/paper.pdf', 'source')
        for name, peer in [('page', 'target'), ('target', 'page')]:
            self.write('wiki/methods/' + name + '.md',
                       '---\ntitle: ' + name + '\ntype: method\ntags:\n  - ptq\n'
                       'sources:\n  - raw/paper.pdf\n---\n# ' + name +
                       '\n[peer](' + peer + '.md)\n## 来源身份\n' + BT + 'https://example.org' + BT + '\n')
        for name in ['INDEX', 'READING']:
            self.write('wiki/' + name + '.md',
                       '---\ntitle: ' + name + '\ntype: index\nsources: []\n---\n'
                       '[page](methods/page.md)\n[target](methods/target.md)\n')
        self.write('wiki/README.md', '# Rules\n')
        self.write('wiki/LOG.md', '# Log\n')
        self.write('README.md', '[wiki](wiki/INDEX.md)\n[raw](raw/source.md)\n[folder](raw/)\n'
                   '[web](https://example.org)\n')
        result = check(self.root, require_raw=True)
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['counts']['knowledge_pages'], 2)
        self.assertEqual(result['counts']['local_sources_checked'], 2)
        self.assertEqual(result['counts']['external_references_skipped'], 1)
        self.write('README.md', '[missing](absent.md)\n')
        result = check(self.root)
        self.assertTrue(any(f['path'] == 'README.md' and f['code'] == 'missing_target'
                            for f in result['findings']))
        self.write('wiki/methods/page.md',
                   (self.root / 'wiki/methods/page.md').read_text() + '[outside](https://example.org)\n')
        self.assertIn('outside_wiki', self.codes(check(self.root)))


if __name__ == '__main__':
    unittest.main()
