import aurel
import pytest

class TestAurelCoreFunctions:
    """Test aurel functions execute"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        Lx, Ly, Lz = 60.0, 40.0, 30.0
        Nx, Ny, Nz = 30, 20, 15
        param = { 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 
                'xmin': -Lx/2, 'ymin': -Ly/2, 'zmin': -Lz/2, 
                'dx': Lx/Nx, 'dy': Ly/Ny, 'dz': Lz/Nz}
        fd = aurel.FiniteDifference(param)
        self.rel = aurel.AurelCore(fd)

    @pytest.mark.parametrize("key", aurel.descriptions)
    def test_executable(self, key):
        output = self.rel[key]