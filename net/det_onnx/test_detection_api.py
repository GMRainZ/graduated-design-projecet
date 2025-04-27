import unittest
import os
from pathlib import Path
from detedctionapi import DetectionAPI
from unittest.mock import patch, MagicMock


class TestDetectionAPI(unittest.TestCase):
    def setUp(self):
        """Initialize the DetectionAPI instance and mock necessary dependencies."""
        self.api = DetectionAPI()
        self.test_data_dir = Path("tests/data")
        os.makedirs(self.test_data_dir, exist_ok=True)

    @patch("detedctionapi.load_model")
    @patch("detedctionapi.pred")
    def test_process_image_batch(self, mock_pred, mock_load_model):
        """Test image batch processing with mocked predictions."""
        # Mock prediction output
        mock_pred.return_value = [
            {"bbox": [100, 100, 50, 50], "confidence": 0.9, "class_id": 0}
        ]

        # Create dummy image files
        dummy_images = [self.test_data_dir / f"image_{i}.jpg" for i in range(3)]
        for img_path in dummy_images:
            with open(img_path, "w") as f:
                f.write("dummy content")

        # Call the method
        all_preds, all_truths = self.api.process_image_batch(
            image_paths=[str(p) for p in dummy_images],
            label_dir=str(self.test_data_dir),
            conf=0.5,
            iou=0.5,
        )

        # Assertions
        self.assertEqual(len(all_preds), 3)
        self.assertEqual(len(all_truths), 3)
        mock_pred.assert_called()

    @patch("detedctionapi.cv2.VideoCapture")
    @patch("detedctionapi.load_model")
    def test_process_video(self, mock_load_model, mock_video_capture):
        """Test video processing with mocked video capture."""
        # Mock video capture and frames
        mock_capture = MagicMock()
        mock_capture.read.side_effect = [
            (True, "frame_data_1"),
            (True, "frame_data_2"),
            (False, None),
        ]
        mock_video_capture.return_value = mock_capture

        # Call the method
        self.api.process_video(
            input_path="dummy_input.mp4",
            output_path="dummy_output.mp4",
            conf=0.5,
            iou=0.5,
        )

        # Assertions
        mock_video_capture.assert_called_with("dummy_input.mp4")
        mock_capture.release.assert_called()

    @patch("argparse.ArgumentParser.parse_args")
    def test_main_function(self, mock_parse_args):
        """Test main function with command-line arguments."""
        mock_parse_args.return_value = MagicMock(
            mode="image",
            image_dir="tests/data/images",
            label_dir="tests/data/labels",
            video_input=None,
            video_output=None,
            conf=0.5,
            iou=0.5,
        )

        # Call the main function
        with patch("detedctionapi.DetectionAPI") as MockAPI:
            from det_onnx_hbb import main
            main()

            # Assertions
            MockAPI.assert_called()
            MockAPI.return_value.process_image_batch.assert_called()


if __name__ == "__main__":
    unittest.main()