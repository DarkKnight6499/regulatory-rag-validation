import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reg_rag.chunking import Passage
from reg_rag.classification.baseline import KeywordClassifier, accuracy, confusion_matrix, macro_f1
from reg_rag.classification.dataset import (GROUP_COLUMN, ID_COLUMN, LABEL_COLUMN, LABELS, SPLIT_COLUMN, TEXT_COLUMN,
                                            build_dataset, group_split, topic_for)
from reg_rag.corpus import (ManifestEntry, ecfr_xml_to_markdown, normalize_guidance_text, read_manifest, sha256_bytes,
                            spaced_hyphens, verify_manifest, write_manifest)
from reg_rag.evalset import load_questions, validate_questions
from reg_rag.retriever import load_passages

ROOT = Path(__file__).resolve().parents[1]
EM = "—"

XML = f"""<?xml version="1.0"?>
<DIV5 N="999" TYPE="PART"><HEAD>PART 999 {EM} TEST PART</HEAD>
<AUTH><HED>Authority:</HED><PSPACE>12 U.S.C. 1</PSPACE></AUTH>
<DIV8 N="999.1" TYPE="SECTION"><HEAD>§ 999.1 Scope.</HEAD>
<P>(a) A firm {EM} any firm {EM} must comply.</P>
<P>(b) <I>Italic</I> text   with   spaces.</P>
<FTNT>1 Footnote words here.</FTNT>
<GPOTABLE><ROW><ENT>A</ENT><ENT>B</ENT></ROW></GPOTABLE>
<CITA>[78 FR 1]</CITA></DIV8></DIV5>"""


class EcfrConversionTests(unittest.TestCase):
    def setUp(self):
        self.md = ecfr_xml_to_markdown(XML)

    def test_headings_become_markdown(self):
        self.assertIn("# PART 999", self.md)
        self.assertIn("### § 999.1 Scope.", self.md)

    def test_paragraph_whitespace_collapsed(self):
        self.assertIn("(b) Italic text with spaces.", self.md)

    def test_authority_and_citation_dropped(self):
        self.assertNotIn("12 U.S.C.", self.md)
        self.assertNotIn("78 FR", self.md)

    def test_footnote_and_table_tagged(self):
        self.assertIn("[footnote] Footnote words here.", self.md)
        self.assertIn("[table] A", self.md)

    def test_no_em_dash_in_output(self):
        self.assertNotIn(EM, self.md)
        self.assertIn("any firm - must", self.md)

    def test_spaced_hyphens_helper(self):
        self.assertEqual(spaced_hyphens(f"a{EM}b"), "a - b")


class GuidanceNormalizeTests(unittest.TestCase):
    RAW = ("preamble junk\nI. INTRODUCTION\nBanks rely on models. Models carry risk.\n"
           "II. VALIDATION\nValidation checks that models\nperform as expected for their use.\nSecond paragraph here.\n")

    def test_sections_tagged_and_preamble_dropped(self):
        out = normalize_guidance_text(self.RAW, "I. INTRODUCTION", r"[IVX]+\. [A-Z ]+", [])
        self.assertNotIn("preamble", out)
        self.assertIn("# I. INTRODUCTION", out)
        self.assertIn("# II. VALIDATION", out)

    def test_wrapped_lines_are_rejoined(self):
        out = normalize_guidance_text(self.RAW, "I. INTRODUCTION", r"[IVX]+\. [A-Z ]+", [])
        self.assertIn("Validation checks that models perform as expected for their use.", out)


class ManifestTests(unittest.TestCase):
    def test_roundtrip_and_hash_check(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            body = "hello corpus\n"
            (d / "a.md").write_bytes(body.encode("utf-8"))
            entry = ManifestEntry("a", "A", "a.md", ["http://x"], "2026-10-01", ["0" * 64], sha256_bytes(body.encode("utf-8")), len(body))
            write_manifest([entry], d / "manifest.json")
            self.assertEqual(read_manifest(d / "manifest.json")[0]["doc_id"], "a")
            self.assertEqual(verify_manifest(d / "manifest.json", d), [])
            (d / "a.md").write_text("tampered\n", encoding="utf-8")
            self.assertTrue(any("mismatch" in p for p in verify_manifest(d / "manifest.json", d)))
            (d / "a.md").unlink()
            self.assertTrue(any("missing" in p for p in verify_manifest(d / "manifest.json", d)))


class ShippedCorpusTests(unittest.TestCase):
    """Integrity checks on the committed corpus and evaluation set (no models needed)."""

    @classmethod
    def setUpClass(cls):
        cls.corpus_dir = ROOT / "data" / "corpus"
        cls.passages = load_passages(cls.corpus_dir)

    def test_manifest_hashes_match_files(self):
        self.assertEqual(verify_manifest(self.corpus_dir / "manifest.json", self.corpus_dir), [])

    def test_all_documents_chunked(self):
        docs = {p.doc_id for p in self.passages}
        self.assertTrue({"12cfr249", "sr26-2", "fr2052a-instructions"} <= docs)

    def test_chunk_ids_unique(self):
        self.assertEqual(len({p.chunk_id for p in self.passages}), len(self.passages))

    def test_citations_nonempty(self):
        self.assertTrue(all(p.citation and p.covers for p in self.passages))

    def test_eval_set_validates_against_corpus(self):
        qs = load_questions(ROOT / "data" / "eval" / "questions.jsonl")
        self.assertEqual(validate_questions(qs, self.passages), [])

    def test_eval_set_has_all_question_types(self):
        qs = load_questions(ROOT / "data" / "eval" / "questions.jsonl")
        self.assertTrue({"answerable", "unanswerable", "adversarial"} <= {q.type for q in qs})
        self.assertGreaterEqual(len(qs), 60)

    def test_adversarial_questions_have_clean_twin_and_payload(self):
        qs = load_questions(ROOT / "data" / "eval" / "questions.jsonl")
        for q in qs:
            if q.type == "adversarial":
                self.assertTrue(q.clean_twin and q.payload)

    def test_no_em_dashes_in_corpus_files(self):
        for f in self.corpus_dir.glob("*.md"):
            self.assertNotIn(EM, f.read_text(encoding="utf-8"), f.name)

    def test_manifest_is_valid_json(self):
        json.loads((self.corpus_dir / "manifest.json").read_text(encoding="utf-8"))


def _passage(doc, section, text, cite="c"):
    return Passage(f"{doc}:{section}:0", doc, cite, section, text, (cite,))


class TopicLabelTests(unittest.TestCase):
    LONG = "x " * 100

    def test_sr_validation_section_labelled_validation(self):
        self.assertEqual(topic_for(_passage("sr26-2", "V. MODEL VALIDATION ", self.LONG)), "model_validation")

    def test_other_sr_sections_are_governance(self):
        self.assertEqual(topic_for(_passage("sr26-2", "VI. GOVERNANCE AND CONTROLS", self.LONG)), "model_governance")

    def test_capital_parts(self):
        self.assertEqual(topic_for(_passage("12cfr217-b", "§ 217.20", self.LONG)), "capital")

    def test_short_and_footnote_text_excluded(self):
        self.assertIsNone(topic_for(_passage("sr26-2", "I", "short")))
        self.assertIsNone(topic_for(_passage("sr26-2", "I", "[footnote] " + self.LONG)))

    def test_fr2052a_is_reporting(self):
        self.assertEqual(topic_for(_passage("fr2052a-instructions", "General", self.LONG)), "reporting")

    def test_labels_come_from_known_set_on_real_corpus(self):
        df = build_dataset(load_passages(ROOT / "data" / "corpus"))
        self.assertTrue(set(df[LABEL_COLUMN]) <= set(LABELS))
        self.assertGreater(len(df), 500)


class GroupSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = build_dataset(load_passages(ROOT / "data" / "corpus"))

    def test_no_group_in_two_splits(self):
        for seed in (0, 1, 2):
            out = group_split(self.df, seed=seed)
            self.assertTrue((out.groupby(GROUP_COLUMN)[SPLIT_COLUMN].nunique() == 1).all())

    def test_every_row_assigned_and_ids_preserved(self):
        out = group_split(self.df)
        self.assertEqual(set(out[SPLIT_COLUMN]), {"train", "val", "test"})
        self.assertEqual(sorted(out[ID_COLUMN]), sorted(self.df[ID_COLUMN]))

    def test_split_is_seed_deterministic(self):
        a, b = group_split(self.df, seed=3), group_split(self.df, seed=3)
        self.assertTrue(a[SPLIT_COLUMN].equals(b[SPLIT_COLUMN]))

    def test_train_is_largest(self):
        counts = group_split(self.df)[SPLIT_COLUMN].value_counts()
        self.assertGreater(counts["train"], counts["val"])
        self.assertGreater(counts["train"], counts["test"])

    def test_tiny_frame_split(self):
        rows = [{ID_COLUMN: str(i), GROUP_COLUMN: f"g{i % 5}", TEXT_COLUMN: "t", LABEL_COLUMN: "a" if i % 2 else "b"} for i in range(20)]
        out = group_split(pd.DataFrame(rows), seed=0)
        self.assertTrue((out.groupby(GROUP_COLUMN)[SPLIT_COLUMN].nunique() == 1).all())


class BaselineAndMetricTests(unittest.TestCase):
    def test_keyword_classifier_picks_topic_by_terms(self):
        clf = KeywordClassifier()
        self.assertEqual(clf.predict_one("Backtesting and outcomes analysis are part of validation."), "model_validation")
        self.assertEqual(clf.predict_one("The report must be submitted with each data element."), "reporting")

    def test_fallback_when_nothing_matches(self):
        self.assertEqual(KeywordClassifier(fallback="capital").predict_one("zzz qqq"), "capital")

    def test_predict_is_deterministic(self):
        clf = KeywordClassifier()
        texts = ["stress test scenario severely adverse", "risk weight for exposures"]
        self.assertEqual(clf.predict(texts), clf.predict(texts))

    def test_accuracy(self):
        self.assertEqual(accuracy(["a", "b"], ["a", "a"]), 0.5)

    def test_confusion_matrix_orientation(self):
        m = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        self.assertEqual(m.tolist(), [[1, 1], [0, 1]])

    def test_macro_f1_perfect_and_worst(self):
        self.assertEqual(macro_f1(["a", "b"], ["a", "b"], ["a", "b"]), 1.0)
        self.assertEqual(macro_f1(["a", "b"], ["b", "a"], ["a", "b"]), 0.0)

    def test_macro_f1_skips_labels_without_support(self):
        self.assertEqual(macro_f1(["a", "a"], ["a", "a"], ["a", "b", "c"]), 1.0)

    def test_hand_review_file_references_real_chunks(self):
        ids = {p.chunk_id for p in load_passages(ROOT / "data" / "corpus")}
        review = pd.read_csv(ROOT / "data" / "classification" / "hand_review.csv")
        self.assertEqual(len(review), 100)
        self.assertTrue(set(review[ID_COLUMN]) <= ids)


if __name__ == "__main__":
    unittest.main()
