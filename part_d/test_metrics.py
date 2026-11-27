"""Small known-answer checks for native-grid ranking, ties and prediction joins."""
import unittest

import numpy as np

from metrics import PixelAUROC, align, frozen_threshold, image_auc, native_map


class EvaluationTests(unittest.TestCase):
    def test_image_and_pixel_auc_handle_ties(self):
        scores = np.array([0, .5, .5, 1], dtype=np.float32)
        labels = np.array([0, 1, 0, 1], dtype=bool)
        pixel = PixelAUROC(4)
        pixel.add(scores[:2].reshape(1, 2), labels[:2].reshape(1, 2))
        pixel.add(scores[2:].reshape(2, 1), labels[2:].reshape(2, 1))
        self.assertEqual(pixel.compute(), .875)
        self.assertEqual(image_auc(labels, scores), .875)
        self.assertEqual(image_auc([0, 1], [2, 2]), .5)

    def test_native_interpolation_preserves_continuous_scores(self):
        field = native_map(np.array([[0, 1]], dtype=np.float32), 4, 2)
        np.testing.assert_array_equal(field[0], [0, .25, .75, 1])
        np.testing.assert_array_equal(field[0], field[1])
        with self.assertRaises(ValueError):
            native_map(np.array([[np.nan]]), 4, 2)

    def test_validation_only_and_sample_id_alignment(self):
        rows=[{"sample_id": k, "split":"val", "label":0} for k in ("a","b")]
        preds=[{"sample_id":"b","score":1}, {"sample_id":"a","score":0}]
        paired = align(rows, preds)
        self.assertEqual([p["score"] for _, p in paired], [0,1])
        self.assertAlmostEqual(frozen_threshold(paired), .99)
        for bad in ([preds[0]], [preds[0],preds[0]], [dict(preds[0], label=1),preds[1]]):
            with self.assertRaises(ValueError):
                align(rows, bad)
        with self.assertRaises(ValueError):
            frozen_threshold([(dict(rows[0], split="test"), preds[1])])


if __name__ == "__main__":
    unittest.main()
