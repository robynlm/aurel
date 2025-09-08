import aurel
import pytest

class TestAurelCoreFunctions:
    """Test aurel functions execute"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        L = 30.0
        N = 39
        param = { 'Nx': N, 'Ny': N, 'Nz': N, 
                'xmin': -L/2, 'ymin': -L/2, 'zmin': -L/2, 
                'dx': L/N, 'dy': L/N, 'dz': L/N}
        fd = aurel.FiniteDifference(param)
        self.rel = aurel.AurelCore(fd)

    @pytest.mark.parametrize("key", aurel.descriptions)
    def test_executable(self, key):
        output = self.rel[key]