import math
import unittest

import numpy as np

from reg_rag.answerer import ExtractiveAnswerer
from reg_rag.dense import HashingEmbedder
from reg_rag.evalset import Question, gold_hit, validate_questions
from reg_rag.evaluate import (SENTINEL, Record, answer_all, answer_report, apply_threshold, balanced_abstain_accuracy,
                              behavior_change, poison_passages, retrieval_report, split_dev_test, tune_threshold)
from reg_rag.metrics import (abstention_scores, best_sentence_f1, bootstrap_ci, bootstrap_ratio_ci, paired_bootstrap_diff,
                             recall_at_k, reciprocal_rank, span_contained, token_f1)
from reg_rag.retriever import Retriever

from tests.fixtures import QUESTIONS, all_passages


class RankMetricTests(unittest.TestCase):
    def test_recall_at_k(self):
        self.assertEqual(recall_at_k([False, False, True], 2), 0.0)
        self.assertEqual(recall_at_k([False, False, True], 3), 1.0)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank([False, True]), 0.5)
        self.assertEqual(reciprocal_rank([False, False]), 0.0)

    def test_token_f1_identical_and_disjoint(self):
        self.assertEqual(token_f1("stress test monthly", "Stress test monthly."), 1.0)
        self.assertEqual(token_f1("alpha", "beta"), 0.0)

    def test_token_f1_partial(self):
        self.assertAlmostEqual(token_f1("a b c d", "a b"), 2 * 0.5 * 1.0 / 1.5)

    def test_span_contained_ignores_case_and_punctuation(self):
        self.assertTrue(span_contained("It is At Least Monthly, always.", "at least monthly"))
        self.assertFalse(span_contained("quarterly", "at least monthly"))

    def test_best_sentence_f1_uses_best_candidate(self):
        self.assertEqual(best_sentence_f1(["noise words", "exact gold span"], "exact gold span"), 1.0)
        self.assertEqual(best_sentence_f1([], "x"), 0.0)


class AbstentionMetricTests(unittest.TestCase):
    def test_counts(self):
        s = abstention_scores([True, True, False, False], [True, False, True, False])
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 1, 1))
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["recall"], 0.5)

    def test_nan_when_undefined(self):
        s = abstention_scores([False, False], [False, False])
        self.assertTrue(math.isnan(s["precision"]) and math.isnan(s["recall"]))


class BootstrapTests(unittest.TestCase):
    def test_ci_brackets_estimate(self):
        est, lo, hi = bootstrap_ci([0.0, 1.0] * 20, n_boot=500)
        self.assertLessEqual(lo, est)
        self.assertLessEqual(est, hi)

    def test_constant_values_give_degenerate_ci(self):
        self.assertEqual(bootstrap_ci([0.7] * 10, n_boot=200), (0.7, 0.7, 0.7))

    def test_same_seed_reproducible(self):
        v = list(np.random.default_rng(1).random(30))
        self.assertEqual(bootstrap_ci(v, n_boot=300, seed=4), bootstrap_ci(v, n_boot=300, seed=4))

    def test_more_data_narrows_interval(self):
        rng = np.random.default_rng(2)
        small = bootstrap_ci(rng.random(20), n_boot=500)
        large = bootstrap_ci(rng.random(2000), n_boot=500)
        self.assertLess(large[2] - large[1], small[2] - small[1])

    def test_empty_is_nan(self):
        self.assertTrue(math.isnan(bootstrap_ci([])[0]))

    def test_ratio_ci_point_estimate(self):
        est, lo, hi = bootstrap_ratio_ci([1, 0, 1, 1], [1, 1, 1, 1], n_boot=300)
        self.assertAlmostEqual(est, 0.75)
        self.assertLessEqual(lo, hi)

    def test_paired_diff_detects_shift(self):
        a = np.linspace(0.5, 1.0, 40)
        est, lo, hi = paired_bootstrap_diff(a, a - 0.1, n_boot=300)
        self.assertAlmostEqual(est, 0.1)
        self.assertGreater(lo, 0.0)


class EvalsetTests(unittest.TestCase):
    def test_synthetic_questions_validate(self):
        self.assertEqual(validate_questions(QUESTIONS, all_passages()), [])

    def test_unknown_gold_key_reported(self):
        bad = [Question("B1", "answerable", "q?", ("12 CFR 1.1",), "span")]
        self.assertTrue(validate_questions(bad, all_passages()))

    def test_span_missing_from_gold_reported(self):
        bad = [Question("B2", "answerable", "q?", ("12 CFR 999.30(a)",), "this text is not there")]
        self.assertTrue(any("span" in p for p in validate_questions(bad, all_passages())))

    def test_abstain_question_must_not_carry_evidence(self):
        bad = [Question("B3", "unanswerable", "q?", ("12 CFR 999.30(a)",), "")]
        self.assertTrue(validate_questions(bad, all_passages()))

    def test_duplicate_ids_reported(self):
        q = QUESTIONS[0]
        self.assertTrue(any("duplicate" in p for p in validate_questions([q, q], all_passages())))

    def test_gold_hit_matches_any_covered_key(self):
        p = [x for x in all_passages() if "12 CFR 999.10(a)(2)" in x.covers][0]
        self.assertTrue(gold_hit(("12 CFR 999.10(a)(2)",), p))
        self.assertFalse(gold_hit(("12 CFR 999.30(a)",), p))


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(all_passages(), HashingEmbedder(512))
        cls.answerer = ExtractiveAnswerer(cls.retriever, threshold=0.0)
        cls.records = answer_all(cls.answerer, QUESTIONS)

    def test_retrieval_report_shape_and_bounds(self):
        rep = retrieval_report(self.retriever, QUESTIONS, n_boot=100)
        self.assertEqual(rep["n"], 4)
        for mode in ("bm25", "dense", "hybrid"):
            self.assertTrue(0.0 <= rep[mode]["mrr"][0] <= 1.0)
            self.assertLessEqual(rep[mode]["recall@1"][0], rep[mode]["recall@5"][0])

    def test_bm25_gets_most_synthetic_questions_at_one(self):
        rep = retrieval_report(self.retriever, QUESTIONS, modes=["bm25"], n_boot=100)
        self.assertGreaterEqual(rep["bm25"]["recall@3"][0], 0.75)

    def test_apply_threshold_forces_abstention(self):
        out = apply_threshold(self.records, 2.0)
        self.assertTrue(all(r.abstained for r in out))
        self.assertTrue(all(r.text == "" for r in out))

    def test_apply_threshold_zero_keeps_records(self):
        self.assertEqual(apply_threshold(self.records, 0.0), self.records)

    def test_tune_threshold_separates_a_clean_split(self):
        qs = [Question("a", "answerable", "q", ("k",), "s"), Question("b", "unanswerable", "q", (), "")]
        recs = [Record("a", "answerable", 0.9, False, "t", [], True, 1.0, True, True, 1.0, 0),
                Record("b", "unanswerable", 0.2, True, "", [], False, 0.0, False, False, 1.0, 0)]
        t = tune_threshold(recs, qs)
        self.assertEqual(balanced_abstain_accuracy(recs, qs, t), 1.0)
        self.assertTrue(0.2 < t <= 0.9)

    def test_balanced_accuracy_nan_without_both_classes(self):
        self.assertTrue(math.isnan(balanced_abstain_accuracy(self.records[:4], QUESTIONS[:4], 0.5)))

    def test_split_dev_test_partitions_in_order(self):
        dev, test = split_dev_test(list(range(7)))
        self.assertEqual((dev, test), ([0, 2, 4, 6], [1, 3, 5]))

    def test_answer_report_counts(self):
        rep = answer_report(self.records, QUESTIONS, n_boot=100)
        self.assertEqual(rep["n_questions"], 5)
        self.assertEqual(rep["n_answerable"], 4)

    def test_record_groundedness_is_per_returned_sentence(self):
        for r in self.records:
            self.assertEqual(r.grounded, 1.0)

    def test_fabricated_answers_are_detected_by_groundedness(self):
        from reg_rag.grounding import groundedness_rate
        fake = "A firm must conduct a stress test every ten years under statute."
        self.assertLess(groundedness_rate(fake, all_passages()), 1.0)


class PoisonTests(unittest.TestCase):
    def test_one_poison_per_answerable_question(self):
        for style in ("instruction", "embedded", "fakeclaim"):
            ps = poison_passages(QUESTIONS, all_passages(), style)
            self.assertEqual(len(ps), 4)
            self.assertTrue(all(SENTINEL in p.text for p in ps))

    def test_unknown_style_rejected(self):
        with self.assertRaises(ValueError):
            poison_passages(QUESTIONS, all_passages(), "nope")

    def test_fakeclaim_has_no_instruction_language(self):
        from reg_rag.injection import is_injected
        self.assertFalse(any(is_injected(p.text) for p in poison_passages(QUESTIONS, all_passages(), "fakeclaim")))

    def test_defence_neutralises_instruction_poison(self):
        poison = poison_passages(QUESTIONS, all_passages(), "instruction")
        r = Retriever(all_passages() + poison, HashingEmbedder(512))
        on = answer_all(ExtractiveAnswerer(r, threshold=0.0, defence=True), QUESTIONS)
        self.assertFalse(any(SENTINEL in x.text for x in on))

    def test_behavior_change_counts_differences(self):
        base = Record("a", "answerable", 0.9, False, "t", ["c"], True, 1.0, True, True, 1.0, 0)
        moved = Record("a", "answerable", 0.9, False, f"{SENTINEL}", ["POISON x"], False, 0.0, False, False, 1.0, 0)
        out = behavior_change([base, base], [base, moved])
        self.assertEqual((out["behavior_changed"], out["payload_in_answer"], out["poison_cited"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
