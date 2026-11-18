import unittest
import numpy as np
from PIL import Image
from imaging import insert_center, raw_center

class ImagingTests(unittest.TestCase):
    def test_only_border_blends(self):
        rng=np.random.default_rng(42)
        context=Image.fromarray(rng.integers(0,256,(1024,1024,3),dtype='uint8'))
        patch=Image.fromarray(rng.integers(0,256,(512,512,3),dtype='uint8'))
        for width in (1,16,32,128):
            out=np.asarray(insert_center(context,patch,width))
            np.testing.assert_array_equal(out[256+width:768-width,256+width:768-width],np.asarray(patch)[width:-width,width:-width])
            outside=np.ones((1024,1024),dtype=bool)
            outside[256:768,256:768]=False
            np.testing.assert_array_equal(out[outside],np.asarray(context)[outside])
            np.testing.assert_array_equal(out[256,256:768],np.asarray(context)[256,256:768])

    def test_zero_width_is_hard_paste(self):
        context=Image.new('RGB',(1024,1024),(20,40,60))
        patch=Image.new('RGB',(512,512),(200,180,160))
        np.testing.assert_array_equal(insert_center(context,patch,0),raw_center(context,patch))

if __name__ == '__main__':
    unittest.main()
