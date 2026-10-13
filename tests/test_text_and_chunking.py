import unittest

from reg_rag.chunking import MAX_CHARS, chunk_cfr_markdown, sentences, split_long
from reg_rag.textproc import normalize_text, stem, tokenize

from tests.fixtures import CFR_MD, cfr_passages, guidance_passages


class TextProcTests(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_stems_plurals(self):
        self.assertEqual(tokenize("The firms are stress testing"), ["firm", "stress", "test"])

    def test_tokenize_keeps_decimal_numbers(self):
        self.assertIn("1.0", tokenize("ratio of 1.0 or more"))

    def test_stem_keeps_short_words_and_numbers(self):
        self.assertEqual(stem("ass"), "ass")
        self.assertEqual(stem("30"), "30")

    def test_normalize_text_strips_punctuation_and_case(self):
        self.assertEqual(normalize_text("Stress, TEST; (a) 1.5%"), "stress test a 1.5")


class SentenceTests(unittest.TestCase):
    def test_sentences_split_on_period_before_capital(self):
        self.assertEqual(len(sentences("First one. Second one. third stays")), 2)

    def test_split_long_respects_max_chars(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(200))
        for piece in split_long(text, 300):
            self.assertLessEqual(len(piece), 300)

    def test_split_long_keeps_all_text(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        self.assertEqual(" ".join(split_long(text, 200)), text)


class CfrChunkingTests(unittest.TestCase):
    def setUp(self):
        self.ps = cfr_passages()
        self.by_id = {p.chunk_id: p for p in self.ps}

    def test_chunk_ids_unique(self):
        self.assertEqual(len({p.chunk_id for p in self.ps}), len(self.ps))

    def test_every_chunk_within_size_bound(self):
        self.assertTrue(all(len(p.text) <= MAX_CHARS * 1.5 for p in self.ps))

    def test_section_citation_keys_present(self):
        covers = {k for p in self.ps for k in p.covers}
        for key in ("12 CFR 999.10(a)", "12 CFR 999.20(b)", "12 CFR 999.30(a)"):
            self.assertIn(key, covers)

    def test_nested_marker_keys_generated(self):
        covers = {k for p in self.ps for k in p.covers}
        self.assertIn("12 CFR 999.10(a)(1)", covers)
        self.assertIn("12 CFR 999.10(a)(2)", covers)

    def test_each_paragraph_text_inside_some_chunk(self):
        joined = " ".join(p.text for p in self.ps)
        self.assertIn("equal to or greater than 1.0", joined)
        self.assertIn("85 percent of fair value", joined)

    def test_footnote_gets_own_chunk_with_key(self):
        fn = [p for p in self.ps if p.text.startswith("[footnote]")]
        self.assertEqual(len(fn), 1)
        self.assertIn("12 CFR 999.30 n.1", fn[0].covers)

    def test_definition_key_uses_term(self):
        covers = {k for p in self.ps for k in p.covers}
        self.assertIn("12 CFR 999.40 [Affiliate]", covers)
        self.assertIn("12 CFR 999.30 [Stress]", covers)

    def test_letter_i_after_h_is_a_letter_not_roman(self):
        covers = {k for p in self.ps for k in p.covers}
        self.assertNotIn("12 CFR 999.20(c)(i)", covers)

    def test_chunks_carry_section_and_doc(self):
        for p in self.ps:
            self.assertEqual(p.doc_id, "synthetic-cfr")
            self.assertIn("999.", p.section_path)

    def test_index_text_prepends_title(self):
        p = self.ps[0]
        self.assertTrue(p.index_text.endswith(p.text))

    def test_em_dash_heading_chunks_without_error(self):
        self.assertGreater(len(chunk_cfr_markdown("x", CFR_MD)), 3)


class GuidanceChunkingTests(unittest.TestCase):
    def test_citations_follow_section_and_paragraph(self):
        cites = [p.citation for p in guidance_passages()]
        self.assertIn("SYN 1-1, I, para 1", cites)
        self.assertTrue(any("Ongoing Monitoring" in c for c in cites))

    def test_covers_include_each_merged_paragraph(self):
        merged = [p for p in guidance_passages() if "Ongoing Monitoring" in p.citation][0]
        self.assertEqual(len(merged.covers), 2)


if __name__ == "__main__":
    unittest.main()
