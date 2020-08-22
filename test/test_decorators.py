import unittest
import numpy as np
from imutils.decorators import imclip, imrange


class TestDecorators(unittest.TestCase):
    """Tester for image decorators."""

    def setUp(self):
        rows, cols, channels = 256, 512, 3
        # Normal range compliant image
        self.img = np.random.rand(rows, cols, channels)
        # Image with some values higher than 1
        self.img_leak_up = np.random.rand(rows, cols, channels)
        self.img_leak_up[self.img_leak_up > 0.5] = \
            self.img_leak_up[self.img_leak_up > 0.5] * 2
        # Image with some values lower than 0
        self.img_leak_down = np.random.rand(rows, cols, channels)
        self.img_leak_down[self.img_leak_down < 0.5] = \
            -self.img_leak_up[self.img_leak_down < 0.5]

    def test_imclip(self):
        """Test imclip decorator."""
        @imclip
        def _return_image(img):
            return img

        # Test that imclip has no effect on compliant images.
        self.assertTrue(np.array_equal(self.img, _return_image(self.img)))
        # Test clipping for images with values higher than 1
        self.assertTrue(np.max(_return_image(self.img_leak_up)) == 1)
        # Test clipping for images with values lower than 0
        self.assertTrue(np.min(_return_image(self.img_leak_down)) == 0)

    def test_imrange(self):
        """Test imrange decorator."""
        @imrange
        def _return_image(img):
            return img

        # Test that imrange has no effect on compliant images.
        self.assertTrue(np.array_equal(self.img, _return_image(self.img)))
        # Test range assertion failure for images with values higher than 1
        with self.assertRaises(AssertionError):
            _return_image(self.img_leak_up)
        # Test range assertion failure for images with values lower than 0
        with self.assertRaises(AssertionError):
            _return_image(self.img_leak_down)


if __name__ == '__main__':
    unittest.main()
