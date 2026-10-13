import unittest

import numpy as np

from reg_rag.bm25 import BM25
from reg_rag.dense import DenseIndex, HashingEmbedder
from reg_rag.retriever import MODES, Retriever, rrf

from tests.fixtures import all_passages


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.docs = ["liquidity ratio minimum requirement", "stress test frequency monthly", "model validation independent review"]
        self.bm = BM25(self.docs)

    def test_matching_doc_ranks_first(self):
        self.assertEqual(self.bm.top("how often is the stress test run", 3)[0], 1)

    def test_unknown_terms_score_zero(self):
        self.assertTrue(np.all(self.bm.scores("zzzz qqqq") == 0))
        self.assertEqual(self.bm.top("zzzz", 3), [])

    def test_rarer_term_has_higher_idf(self):
        bm = BM25(["alpha beta", "alpha gamma", "alpha delta"])
        self.assertGreater(bm.idf["beta"], bm.idf["alpha"])

    def test_scores_nonnegative_and_sized(self):
        s = self.bm.scores("liquidity model")
        self.assertEqual(len(s), 3)
        self.assertTrue((s >= 0).all())

    def test_repeated_term_saturates(self):
        bm = BM25(["cat " * 2 + "x", "cat " * 40 + "x", "dog y z"])
        s = bm.scores("cat")
        self.assertLess(s[1] / s[0], 20)

    def test_top_k_limit(self):
        self.assertLessEqual(len(self.bm.top("liquidity stress model", 2)), 2)


class HashingEmbedderTests(unittest.TestCase):
    def test_deterministic(self):
        e = HashingEmbedder(128)
        a, b = e.encode(["stress test monthly"]), e.encode(["stress test monthly"])
        self.assertTrue(np.array_equal(a, b))

    def test_unit_norm(self):
        v = HashingEmbedder(128).encode(["liquidity coverage ratio"])
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 1.0, places=5)

    def test_similar_text_closer_than_unrelated(self):
        e = HashingEmbedder(512)
        v = e.encode(["stress test frequency", "frequency of the stress test", "model validation review"])
        self.assertGreater(float(v[0] @ v[1]), float(v[0] @ v[2]))

    def test_empty_text_gives_zero_vector_not_nan(self):
        v = HashingEmbedder(64).encode([""])
        self.assertFalse(np.isnan(v).any())


class RRFTests(unittest.TestCase):
    def test_agreement_beats_single_vote(self):
        fused = rrf([[1, 2, 3], [1, 3, 2]])
        self.assertEqual(fused[0][0], 1)

    def test_scores_follow_formula(self):
        fused = dict(rrf([[7]], k=60))
        self.assertAlmostEqual(fused[7], 1 / 61)

    def test_union_of_rankings(self):
        self.assertEqual({d for d, _ in rrf([[1, 2], [3]])}, {1, 2, 3})

    def test_ties_break_by_id(self):
        self.assertEqual([d for d, _ in rrf([[5], [4]])], [4, 5])


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Retriever(all_passages(), HashingEmbedder(512))

    def test_all_modes_return_k_hits(self):
        for mode in MODES:
            self.assertEqual(len(self.r.search("stress test frequency", k=3, mode=mode)), 3)

    def test_bm25_finds_gold_passage(self):
        top = self.r.search("How often must a firm conduct a stress test?", k=1, mode="bm25")[0]
        self.assertIn("12 CFR 999.30(a)", top.passage.covers)

    def test_hits_are_ranked_from_one(self):
        hits = self.r.search("liquidity ratio", k=4, mode="hybrid")
        self.assertEqual([h.rank for h in hits], [1, 2, 3, 4])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.r.search("x", mode="magic")

    def test_dense_index_scores_cosine_range(self):
        idx = DenseIndex(["alpha beta", "gamma delta"], HashingEmbedder(64))
        s = idx.scores("alpha")
        self.assertTrue(((s >= -1.0001) & (s <= 1.0001)).all())


if __name__ == "__main__":
    unittest.main()
