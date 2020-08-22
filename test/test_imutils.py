import unittest
import numpy as np
from unittest.mock import patch

from imutils import (
    imbright,
    imcrop,
    imgamma,
    imnoise,
    imread,
    random_number,
)


# Dummy image to avoid loading from disk
loaded_image = (np.random.rand(256, 256, 3)*255).astype(np.uint8)


class TestDetection(unittest.TestCase):
    """Tester for image proicessing utilities."""

    def setUp(self):
        self.img = np.random.rand(256, 256, 3)
        self.mean = np.mean(self.img)
        self.epsilon = 1e-6

    def assertClip(self, img):
        """Asserts image in [0, 1] interval."""
        self.assertTrue(np.min(img) >= 0 and np.max(img) <= 1.0)

    def test_imbright(self):
        """Test imbright utility."""
        # Check exact shift
        shifts = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        _img = np.copy(self.img)
        _img[_img < 0.3] = 0.3
        _img[_img > 0.7] = 0.7
        imgs = [imbright(_img, shift=sh, random=False) for sh in shifts]
        for sh, img in zip(shifts, imgs):
            diff = np.abs(img - (_img + sh))
            self.assertTrue(np.mean(diff) < self.epsilon)
            self.assertClip(img)

        # Check random shift
        max_shift = 0.3
        imgs = [imbright(self.img, shift=max_shift, random=True)
                for _ in range(100)]
        for img in imgs:
            self.assertTrue(np.mean(img) <= self.mean + max_shift)
            self.assertTrue(np.mean(img) >= self.mean - max_shift)
            self.assertClip(img)

    def test_imgamma(self):
        """Test imgamma utility."""
        # Test corrections for different gammas
        gammas = [0.5, 0.7, 0.9, 1.1, 2, 3]
        imgs = [imgamma(self.img, gamma=gamma, random=False)
                for gamma in gammas]

        for gamma, img in zip(gammas, imgs):
            self.assertClip(img)
            if gamma > 1:
                self.assertTrue(np.mean(img) < self.mean)
                self.assertTrue(np.all((img - self.img) <= 0))
            else:
                self.assertTrue(np.mean(img) > self.mean)
                self.assertTrue(np.all((img - self.img) >= 0))

        # Test correction for random gammas
        imgs = [imgamma(self.img, gamma=2, random=True) for _ in range(100)]
        for img in imgs:
            self.assertTrue(np.all(img - self.img > 0) or \
                            np.all(img - self.img < 0))

    def test_imnoise(self):
        """Test imnoise utility."""
        # Check fixed std
        stds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        imgs = [imnoise(self.img, std=std, random=False) for std in stds]
        for img in imgs:
            self.assertClip(img)
            self.assertTrue(
                np.allclose(np.mean(img) - self.mean, 0, atol=1e-2)
                )

        # Check random std
        imgs = [imnoise(self.img, std=0.5, random=True) for _ in range(100)]
        for img in imgs:
            self.assertClip(img)
            self.assertTrue(
                np.allclose(np.mean(img) - self.mean, 0, atol=1e-2)
                )

    def test_random_number(self):
        """Test random number generator."""
        # Unsigned number from interval [0,1]
        x = np.array([random_number(sign=False) for _ in range(1000)])
        self.assertTrue(np.min(x) >= 0 and np.max(x) <= 1)

        # Signed
        x = np.array([random_number(sign=True) for _ in range(1000)])
        self.assertTrue(np.min(x) >= -1 and np.max(x) <= 1)
        self.assertTrue(np.sum(x > 0) / np.sum(x < 0) > 0.95)

    @patch('imutils.cv2.imread', return_value=loaded_image)
    def test_imread(self, mock_imread):
        """Test imread utility."""
        img = imread(filename=None, standardize=False)
        self.assertClip(img)
        self.assertTrue(img.shape == loaded_image.shape)

        img = imread(filename=None, standardize=True)
        self.assertTrue(np.abs(np.mean(img)) < self.epsilon)
        self.assertTrue(np.abs(np.std(img) - 1.0) < self.epsilon)
        self.assertTrue(img.shape == loaded_image.shape)

    def test_imcrop(self):
        """Test imcrop utility."""
        # No imcrop effect on same size target
        img = imcrop(self.img, size=(self.img.shape[0], self.img.shape[1]))
        self.assertTrue(img.shape == self.img.shape)

        for _ in range(0, 100):
            # Test imcrop on color images
            img = imcrop(self.img, size=(200, 200))
            self.assertTrue(img.shape == (200, 200, 3))
            # Test imcrop on monochromatic images
            img = imcrop(self.img[:, :, 0], size=(200, 200))
            self.assertTrue(img.shape == (200, 200))


if __name__ == '__main__':
    unittest.main()
