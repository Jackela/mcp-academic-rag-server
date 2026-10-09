"""Local numeric contracts for vectors, grouping, filtering and optional callbacks."""

import unittest

from utils.vector_utils import (
    combine_embeddings,
    evaluate_embeddings,
    filter_embeddings,
    get_academic_citation_embeddings,
    optimize_chunk_size,
)


class VectorUtilityTests(unittest.TestCase):
    def test_weighted_integer_vectors_produce_floating_average(self):
        self.assertEqual(combine_embeddings([[1, 0], [0, 1]], method="weighted"), [0.5, 0.5])

    def test_grouped_coherence_separates_orthogonal_classes(self):
        self.assertEqual(
            evaluate_embeddings([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], ["a", "a", "b", "b"]),
            {"intra_class_similarity": 1.0, "inter_class_distance": 1.0, "separation_ratio": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "不匹配"):
            evaluate_embeddings([[1.0]], [])

    def test_filtering_retains_order_and_text_alignment(self):
        vectors = [[1.0, 0.0], [1.1, 0.0], [50.0, 0.0]]
        self.assertEqual(
            filter_embeddings(vectors, ["first", "second", "outlier"], threshold=1.0),
            (vectors[:2], ["first", "second"], [0, 1]),
        )
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(
            filter_embeddings(vectors, ["first", "second", "noise"], "noise", 0.4),
            (vectors[:2], ["first", "second"], [0, 1]),
        )

    def test_optional_embedding_callback_remains_optional(self):
        self.assertEqual(get_academic_citation_embeddings("A source [1]"), ([], []))
        size, score = optimize_chunk_size("A sentence. " * 50, min_size=100, max_size=200)
        self.assertIn(size, [100, 200])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
