import unittest
from utils import *

class TestMatrices(unittest.TestCase):

    # test that the sums of CF's rows is equal to CF_SCALE
    def test_CF_rows(self):
        CF = np.load('files/CF.npy', allow_pickle=True)
        
        invalid = False
        for i in range(C):
            if sum(CF[i]) != CF_SCALE:
                invalid = True
            break

        self.assertFalse(invalid)

    # test that the sums of FC's rows is equal to FC_SCALE
    def test_FC(self):
        FC = np.load('files/FC.npy', allow_pickle=True)
        
        invalid = False
        for i in range(F):
            if sum(FC[i]) != FC_SCALE:
                invalid = True
            break

        self.assertFalse(invalid)

if __name__ == '__main__':
    unittest.main()
