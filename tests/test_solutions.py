"""
Tests for the solutions modules.

These tests verify that all solution modules can be imported and their
functions can be called without errors, and that they return properly shaped outputs.
"""

import numpy as np
import pytest
import sympy as sp

from aurel.solutions import (
    LCDM,
    Collins_Stewart,
    Conformally_flat,
    EdS,
    Harvey_Tsoubelis,
    ICPertFLRW,
    Non_diagonal,
    Rosquist_Jantzen,
    Schwarzschild_isotropic,
    Szekeres,
)

# =============================================================================
#                           Test Grid Setup
# =============================================================================

@pytest.fixture(scope="module")
def test_grid():
    """Create a small test grid for solution functions."""
    N = 8
    xmin, xmax = -1.0, 1.0
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(xmin, xmax, N)
    z = np.linspace(xmin, xmax, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    t = 1.0  # test time
    return t, X, Y, Z


# =============================================================================
#                      Test Collins & Stewart Solution
# =============================================================================

class TestCollinsStewart:
    """Tests for Collins_Stewart solution."""

    def test_imports(self):
        """Test that module constants are defined."""
        assert hasattr(Collins_Stewart, 'kappa')
        assert hasattr(Collins_Stewart, 'gamma')

    def test_rho(self, test_grid):
        """Test energy density function."""
        t, x, y, z = test_grid
        rho = Collins_Stewart.rho(t, x, y, z)
        # Returns scalar value for this solution
        assert isinstance(rho, (float, np.floating))
        assert not np.isnan(rho)

    def test_press(self, test_grid):
        """Test pressure function."""
        t, x, y, z = test_grid
        press = Collins_Stewart.press(t, x, y, z)
        # Returns scalar value for this solution
        assert isinstance(press, (float, np.floating))
        assert not np.isnan(press)

    def test_gammadown3(self, test_grid):
        """Test spatial metric."""
        t, x, y, z = test_grid
        gamma = Collins_Stewart.gammadown3(t, x, y, z)
        assert gamma.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(gamma))

    def test_gammadown3_analytical(self, test_grid):
        """Test spatial metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        gamma = Collins_Stewart.gammadown3(t, x, y, z, analytical=True)
        assert gamma.shape == (3, 3)
        # Should return SymPy Matrix
        assert isinstance(gamma, sp.Matrix)
        # Check that elements are SymPy expressions
        assert isinstance(gamma[0, 0], sp.Expr)

    def test_gdown4_analytical(self):
        """Test spacetime metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        g = Collins_Stewart.gdown4(t, x, y, z, analytical=True)
        assert g.shape == (4, 4)
        assert isinstance(g, sp.Matrix)
        # Check metric signature element
        assert g[0, 0] == -1

    def test_Kdown3(self, test_grid):
        """Test extrinsic curvature."""
        t, x, y, z = test_grid
        K = Collins_Stewart.Kdown3(t, x, y, z)
        assert K.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(K))

    def test_gdown4(self, test_grid):
        """Test spacetime metric."""
        t, x, y, z = test_grid
        g = Collins_Stewart.gdown4(t, x, y, z)
        assert g.shape == (4, 4) + x.shape
        assert not np.any(np.isnan(g))

    def test_data_function(self, test_grid):
        """Test that data() returns complete dictionary."""
        t, x, y, z = test_grid
        data = Collins_Stewart.data(t, x, y, z)
        assert 'gammadown3' in data
        assert 'rho' in data
        assert 'press' in data
        assert 'Kdown3' in data


# =============================================================================
#                      Test Conformally Flat Solution
# =============================================================================

class TestConformallyFlat:
    """Tests for Conformally_flat solution."""

    def test_imports(self):
        """Test that module constants are defined."""
        assert hasattr(Conformally_flat, 'eps')
        assert hasattr(Conformally_flat, 'kappa')

    def test_alpha(self, test_grid):
        """Test lapse function."""
        t, x, y, z = test_grid
        alpha = Conformally_flat.alpha(t, x, y, z)
        assert alpha.shape == x.shape
        assert not np.any(np.isnan(alpha))

    def test_gammadown3(self, test_grid):
        """Test spatial metric."""
        t, x, y, z = test_grid
        gamma = Conformally_flat.gammadown3(t, x, y, z)
        assert gamma.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(gamma))

    def test_Kdown3(self, test_grid):
        """Test extrinsic curvature."""
        t, x, y, z = test_grid
        K = Conformally_flat.Kdown3(t, x, y, z)
        assert K.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(K))

    def test_data_function(self, test_grid):
        """Test that data() returns complete dictionary."""
        t, x, y, z = test_grid
        data = Conformally_flat.data(t, x, y, z)
        assert 'gammadown3' in data
        assert 'alpha' in data
        assert 'Tdown4' in data


# =============================================================================
#                      Test Einstein-de Sitter Solution
# =============================================================================

class TestEdS:
    """Tests for EdS (Einstein-de Sitter) solution."""

    def test_imports(self):
        """Test that module constants are defined."""
        assert hasattr(EdS, 'h')
        assert hasattr(EdS, 'Omega_m_EdS')
        assert hasattr(EdS, 'kappa')

    def test_a(self):
        """Test scale factor."""
        t = 1.0
        a = EdS.a(t)
        assert isinstance(a, (float, np.ndarray))
        assert a > 0

    def test_Hprop(self):
        """Test Hubble function."""
        t = 1.0
        H = EdS.Hprop(t)
        assert isinstance(H, (float, np.ndarray))
        assert H > 0

    def test_redshift(self):
        """Test redshift function."""
        t = EdS.t_today
        z = EdS.redshift(t)
        assert np.isclose(z, 0.0, atol=1e-10)  # z=0 at t_today

    def test_rho(self):
        """Test energy density."""
        t = 1.0
        rho = EdS.rho(t)
        assert rho > 0

    def test_alpha(self, test_grid):
        """Test lapse function."""
        t, x, y, z = test_grid
        alpha = EdS.alpha(t, x, y, z)
        assert alpha.shape == x.shape
        assert np.all(alpha == 1.0)  # Should be unity

    def test_betaup3(self, test_grid):
        """Test shift vector."""
        t, x, y, z = test_grid
        beta = EdS.betaup3(t, x, y, z)
        assert beta.shape == (3,) + x.shape
        assert np.all(beta == 0.0)  # Should be zero


# =============================================================================
#                      Test Harvey & Tsoubelis Solution
# =============================================================================

class TestHarveyTsoubelis:
    """Tests for Harvey_Tsoubelis solution."""

    def test_imports(self):
        """Test that module can be imported."""
        assert Harvey_Tsoubelis is not None

    def test_rho(self, test_grid):
        """Test energy density function."""
        t, x, y, z = test_grid
        rho = Harvey_Tsoubelis.rho(t, x, y, z)
        assert rho.shape == x.shape
        assert not np.any(np.isnan(rho))

    def test_gammadown3(self, test_grid):
        """Test spatial metric."""
        t, x, y, z = test_grid
        gamma = Harvey_Tsoubelis.gammadown3(t, x, y, z)
        assert gamma.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(gamma))

    def test_Kdown3(self, test_grid):
        """Test extrinsic curvature."""
        t, x, y, z = test_grid
        K = Harvey_Tsoubelis.Kdown3(t, x, y, z)
        assert K.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(K))


# =============================================================================
#                      Test ICPertFLRW Solution
# =============================================================================

class TestICPertFLRW:
    """Tests for ICPertFLRW (perturbed FLRW) solution.

    Note: This module has a different API - functions take sol and fd objects.
    """

    def test_imports(self):
        """Test that module can be imported and has expected functions."""
        assert hasattr(ICPertFLRW, 'gammadown3')
        assert hasattr(ICPertFLRW, 'Kdown3')
        assert hasattr(ICPertFLRW, 'delta1')
        assert hasattr(ICPertFLRW, 'Rc_func')

    def test_Rc_func(self, test_grid):
        """Test comoving curvature perturbation function."""
        t, x, y, z = test_grid
        amp = (0.01, 0.01, 0.01)
        lamb = (10.0, 10.0, 10.0)
        Rc = ICPertFLRW.Rc_func(x, y, z, amp, lamb)
        assert Rc.shape == x.shape
        assert not np.any(np.isnan(Rc))
        # Check that perturbation is small
        assert np.abs(Rc).max() < 0.1

    def test_gammadown3_with_mock_objects(self, test_grid):
        """Test spatial metric with mock sol and fd objects."""
        import aurel
        from aurel.solutions import EdS

        t, x, y, z = test_grid
        # Create a FiniteDifference object
        param = {
            'xmin': -1.0, 'xmax': 1.0, 'Nx': 8, 'dx': 2.0/7,
            'ymin': -1.0, 'ymax': 1.0, 'Ny': 8, 'dy': 2.0/7,
            'zmin': -1.0, 'zmax': 1.0, 'Nz': 8, 'dz': 2.0/7,
        }
        fd = aurel.FiniteDifference(param)

        # Create perturbation field
        amp = (0.001, 0.001, 0.001)
        lamb = (2.0, 2.0, 2.0)
        Rc = ICPertFLRW.Rc_func(x, y, z, amp, lamb)

        # Use EdS as background solution
        gamma = ICPertFLRW.gammadown3(EdS, fd, 1.0, Rc)
        assert gamma.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(gamma))

    def test_Kdown3_with_mock_objects(self, test_grid):
        """Test extrinsic curvature with mock sol and fd objects."""
        import aurel
        from aurel.solutions import EdS

        t, x, y, z = test_grid
        param = {
            'xmin': -1.0, 'xmax': 1.0, 'Nx': 8, 'dx': 2.0/7,
            'ymin': -1.0, 'ymax': 1.0, 'Ny': 8, 'dy': 2.0/7,
            'zmin': -1.0, 'zmax': 1.0, 'Nz': 8, 'dz': 2.0/7,
        }
        fd = aurel.FiniteDifference(param)

        amp = (0.001, 0.001, 0.001)
        lamb = (2.0, 2.0, 2.0)
        Rc = ICPertFLRW.Rc_func(x, y, z, amp, lamb)

        K = ICPertFLRW.Kdown3(EdS, fd, 1.0, Rc)
        assert K.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(K))

    def test_delta1_with_mock_objects(self, test_grid):
        """Test linear density contrast with mock sol and fd objects."""
        import aurel
        from aurel.solutions import EdS

        t, x, y, z = test_grid
        param = {
            'xmin': -1.0, 'xmax': 1.0, 'Nx': 8, 'dx': 2.0/7,
            'ymin': -1.0, 'ymax': 1.0, 'Ny': 8, 'dy': 2.0/7,
            'zmin': -1.0, 'zmax': 1.0, 'Nz': 8, 'dz': 2.0/7,
        }
        fd = aurel.FiniteDifference(param)

        amp = (0.001, 0.001, 0.001)
        lamb = (2.0, 2.0, 2.0)
        Rc = ICPertFLRW.Rc_func(x, y, z, amp, lamb)

        delta = ICPertFLRW.delta1(EdS, fd, 1.0, Rc)
        assert delta.shape == x.shape
        assert not np.any(np.isnan(delta))


# =============================================================================
#                      Test LCDM Solution
# =============================================================================

class TestLCDM:
    """Tests for LCDM (Lambda CDM) solution."""

    def test_imports(self):
        """Test that module constants are defined."""
        assert hasattr(LCDM, 'h')
        assert hasattr(LCDM, 'Omega_m_today')
        assert hasattr(LCDM, 'Lambda')

    def test_a(self):
        """Test scale factor."""
        t = 1.0
        a = LCDM.a(t)
        assert isinstance(a, (float, np.ndarray))
        assert a > 0

    def test_Hprop(self):
        """Test Hubble function."""
        t = 1.0
        H = LCDM.Hprop(t)
        assert isinstance(H, (float, np.ndarray))
        assert H > 0

    def test_redshift(self):
        """Test redshift function exists."""
        assert hasattr(LCDM, 'redshift')
        assert callable(LCDM.redshift)

    def test_rho(self):
        """Test energy density."""
        t = 1.0
        rho = LCDM.rho(t)
        assert rho > 0

    def test_alpha(self, test_grid):
        """Test lapse function."""
        t, x, y, z = test_grid
        alpha = LCDM.alpha(t, x, y, z)
        assert alpha.shape == x.shape
        assert np.all(alpha == 1.0)


# =============================================================================
#                      Test Non-diagonal Solution
# =============================================================================

class TestNonDiagonal:
    """Tests for Non_diagonal solution."""

    def test_imports(self):
        """Test that module constants and functions are defined."""
        assert hasattr(Non_diagonal, 'kappa')
        assert hasattr(Non_diagonal, 'Lambda')
        assert hasattr(Non_diagonal, 'fq')

    def test_A(self, test_grid):
        """Test conformal factor function."""
        t, x, y, z = test_grid
        A_val = Non_diagonal.A(z)
        assert A_val.shape == z.shape
        assert not np.any(np.isnan(A_val))
        # Check reasonable range
        assert np.all(A_val > 2.0) and np.all(A_val < 2.6)

    def test_A_analytical(self):
        """Test conformal factor in analytical mode."""
        import sympy as sp
        z_sym = sp.Symbol('z')
        A_val = Non_diagonal.A(z_sym, analytical=True)
        assert A_val is not None
        # Should be a SymPy expression
        assert hasattr(A_val, 'subs')

    def test_dzA(self, test_grid):
        """Test conformal factor first derivative."""
        t, x, y, z = test_grid
        dzA = Non_diagonal.dzA(z)
        assert dzA.shape == z.shape
        assert not np.any(np.isnan(dzA))

    def test_dzdzA(self, test_grid):
        """Test conformal factor second derivative."""
        t, x, y, z = test_grid
        dzdzA = Non_diagonal.dzdzA(z)
        assert dzdzA.shape == z.shape
        assert not np.any(np.isnan(dzdzA))

    def test_gammadown3(self, test_grid):
        """Test spatial metric."""
        t, x, y, z = test_grid
        gamma = Non_diagonal.gammadown3(t, x, y, z)
        assert gamma.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(gamma))
        # Check non-diagonal structure - off-diagonal elements should be non-zero
        assert not np.all(gamma[0, 1] == 0.0)  # gxy component

    def test_gammadown3_analytical(self):
        """Test spatial metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        gamma = Non_diagonal.gammadown3(t, x, y, z, analytical=True)
        assert gamma.shape == (3, 3)
        # Should be SymPy Matrix
        assert isinstance(gamma, sp.Matrix)
        assert hasattr(gamma, 'det')
        # Check non-diagonal structure
        assert gamma[0, 1] != 0

    def test_gdown4(self, test_grid):
        """Test spacetime metric."""
        t, x, y, z = test_grid
        g = Non_diagonal.gdown4(t, x, y, z)
        assert g.shape == (4, 4) + x.shape
        assert not np.any(np.isnan(g))
        # Check signature: g00 should be negative
        assert np.all(g[0, 0] < 0)

    def test_gdown4_analytical(self):
        """Test spacetime metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        g = Non_diagonal.gdown4(t, x, y, z, analytical=True)
        assert g.shape == (4, 4)
        assert isinstance(g, sp.Matrix)
        # Check metric signature
        assert g[0, 0] == -1

    def test_Kdown3(self, test_grid):
        """Test extrinsic curvature."""
        t, x, y, z = test_grid
        K = Non_diagonal.Kdown3(t, x, y, z)
        assert K.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(K))
        # Diagonal components should be non-zero
        assert not np.all(K[0, 0] == 0.0)

    def test_Tdown4(self, test_grid):
        """Test stress-energy tensor."""
        t, x, y, z = test_grid
        T = Non_diagonal.Tdown4(t, x, y, z)
        assert T.shape == (4, 4) + x.shape
        assert not np.any(np.isnan(T))
        # Energy density (T00) should be positive
        assert np.all(T[0, 0] > 0)

    def test_data_function(self, test_grid):
        """Test that data() returns complete dictionary."""
        t, x, y, z = test_grid
        data = Non_diagonal.data(t, x, y, z)
        assert 'gammadown3' in data
        assert 'Kdown3' in data
        assert 'Tdown4' in data
        # Verify shapes
        assert data['gammadown3'].shape == (3, 3) + x.shape
        assert data['Kdown3'].shape == (3, 3) + x.shape
        assert data['Tdown4'].shape == (4, 4) + x.shape


# =============================================================================
#                      Test Rosquist & Jantzen Solution
# =============================================================================

class TestRosquistJantzen:
    """Tests for Rosquist_Jantzen solution."""

    def test_imports(self):
        """Test that module can be imported and has constants defined."""
        assert Rosquist_Jantzen is not None
        assert hasattr(Rosquist_Jantzen, 'kappa')
        assert hasattr(Rosquist_Jantzen, 'gamma')
        assert hasattr(Rosquist_Jantzen, 'k')
        assert hasattr(Rosquist_Jantzen, 'm')

    def test_gammadown3(self, test_grid):
        """Test spatial metric."""
        t, x, y, z = test_grid
        gamma = Rosquist_Jantzen.gammadown3(t, x, y, z)
        assert gamma.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(gamma))
        # Check non-diagonal structure (should have off-diagonal xy component)
        assert not np.all(gamma[0, 1] == 0.0)
        # Check symmetry
        assert np.allclose(gamma[0, 1], gamma[1, 0])

    def test_gammadown3_analytical(self):
        """Test spatial metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        gamma = Rosquist_Jantzen.gammadown3(t, x, y, z, analytical=True)
        assert gamma.shape == (3, 3)
        assert isinstance(gamma, sp.Matrix)
        # Check non-diagonal structure
        assert gamma[0, 1] != 0
        # Check symmetry
        assert gamma[0, 1] == gamma[1, 0]

    def test_Kdown3(self, test_grid):
        """Test extrinsic curvature."""
        t, x, y, z = test_grid
        K = Rosquist_Jantzen.Kdown3(t, x, y, z)
        assert K.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(K))
        # Diagonal components should be non-zero
        assert not np.all(K[0, 0] == 0.0)
        assert not np.all(K[1, 1] == 0.0)
        assert not np.all(K[2, 2] == 0.0)
        # Check symmetry
        assert np.allclose(K[0, 1], K[1, 0])

    def test_gdown4(self, test_grid):
        """Test spacetime metric."""
        t, x, y, z = test_grid
        g = Rosquist_Jantzen.gdown4(t, x, y, z)
        assert g.shape == (4, 4) + x.shape
        assert not np.any(np.isnan(g))
        # Check metric signature: g00 should be negative
        assert np.all(g[0, 0] < 0)

    def test_gdown4_analytical(self):
        """Test spacetime metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        g = Rosquist_Jantzen.gdown4(t, x, y, z, analytical=True)
        assert g.shape == (4, 4)
        assert isinstance(g, sp.Matrix)
        # Check metric signature
        assert g[0, 0] == -1

    def test_Tdown4(self, test_grid):
        """Test stress-energy tensor."""
        t, x, y, z = test_grid
        T = Rosquist_Jantzen.Tdown4(t, x, y, z)
        assert T.shape == (4, 4) + x.shape
        assert not np.any(np.isnan(T))
        # Energy density (T00) should be positive
        assert np.all(T[0, 0] > 0)

    def test_data_function(self, test_grid):
        """Test that data() returns complete dictionary."""
        t, x, y, z = test_grid
        data = Rosquist_Jantzen.data(t, x, y, z)
        assert 'gammadown3' in data
        assert 'Kdown3' in data
        assert 'Tdown4' in data
        # Verify shapes
        assert data['gammadown3'].shape == (3, 3) + x.shape
        assert data['Kdown3'].shape == (3, 3) + x.shape
        assert data['Tdown4'].shape == (4, 4) + x.shape


# =============================================================================
#                      Test Schwarzschild Solution
# =============================================================================

class TestSchwarzschildIsotropic:
    """Tests for Schwarzschild_isotropic solution."""

    def test_imports(self):
        """Test that module constants are defined."""
        assert hasattr(Schwarzschild_isotropic, 'kappa')
        assert hasattr(Schwarzschild_isotropic, 'M')

    def test_alpha(self, test_grid):
        """Test lapse function."""
        t, x, y, z = test_grid
        alpha = Schwarzschild_isotropic.alpha(t, x, y, z)
        assert alpha.shape == x.shape
        assert not np.any(np.isnan(alpha))

    def test_alpha_analytical(self):
        """Test lapse function in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        alpha = Schwarzschild_isotropic.alpha(t, x, y, z, analytical=True)
        assert alpha is not None
        # Should be a SymPy expression
        assert isinstance(alpha, sp.Expr)

    def test_gammadown3_analytical(self):
        """Test spatial metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        gamma = Schwarzschild_isotropic.gammadown3(t, x, y, z, analytical=True)
        assert gamma.shape == (3, 3)
        assert isinstance(gamma, sp.Matrix)
        # Should be conformally flat (diagonal)
        assert gamma[0, 1] == 0
        assert gamma[0, 2] == 0
        assert gamma[1, 2] == 0

    def test_gdown4_analytical(self):
        """Test spacetime metric in analytical mode."""
        t, x, y, z = sp.symbols('t x y z')
        g = Schwarzschild_isotropic.gdown4(t, x, y, z, analytical=True)
        assert g.shape == (4, 4)
        assert isinstance(g, sp.Matrix)
        # Check off-diagonal time-space elements are zero
        assert g[0, 1] == 0
        assert g[0, 2] == 0
        assert g[0, 3] == 0

    def test_betaup3(self, test_grid):
        """Test shift vector."""
        t, x, y, z = test_grid
        beta = Schwarzschild_isotropic.betaup3(t, x, y, z)
        assert beta.shape == (3,) + x.shape
        assert np.all(beta == 0.0)

    def test_gammadown3(self, test_grid):
        """Test spatial metric."""
        t, x, y, z = test_grid
        gamma = Schwarzschild_isotropic.gammadown3(t, x, y, z)
        assert gamma.shape == (3, 3) + x.shape
        assert not np.any(np.isnan(gamma))

    def test_Kdown3(self, test_grid):
        """Test extrinsic curvature."""
        t, x, y, z = test_grid
        K = Schwarzschild_isotropic.Kdown3(t, x, y, z)
        assert K.shape == (3, 3) + x.shape
        assert np.all(K == 0.0)  # Zero for static solution

    def test_Tdown4(self, test_grid):
        """Test stress-energy tensor."""
        t, x, y, z = test_grid
        T = Schwarzschild_isotropic.Tdown4(t, x, y, z)
        assert T.shape == (4, 4) + x.shape
        assert np.all(T == 0.0)  # Zero for vacuum

    def test_Kretschmann(self, test_grid):
        """Test Kretschmann scalar."""
        t, x, y, z = test_grid
        K = Schwarzschild_isotropic.Kretschmann(t, x, y, z)
        assert K.shape == x.shape
        assert not np.any(np.isnan(K))

    def test_data_function(self, test_grid):
        """Test that data() returns complete dictionary."""
        t, x, y, z = test_grid
        data = Schwarzschild_isotropic.data(t, x, y, z)
        assert 'alpha' in data
        assert 'gammadown3' in data
        assert 'Kdown3' in data


# =============================================================================
#                      Test Szekeres Solution
# =============================================================================

class TestSzekeres:
    """Tests for Szekeres solution."""

    def test_imports(self):
        """Test that module can be imported."""
        assert hasattr(Szekeres, 'Amp')
        assert hasattr(Szekeres, 'L')

    def test_rho(self, test_grid):
        """Test energy density function."""
        t, x, y, z = test_grid
        rho = Szekeres.rho(t, x, y, z)
        assert rho.shape == x.shape
        assert not np.any(np.isnan(rho))

    def test_press(self, test_grid):
        """Test pressure function."""
        t, x, y, z = test_grid
        press = Szekeres.press(t, x, y, z)
        assert press.shape == x.shape
        # Pressure can be zero for dust


# =============================================================================
#                      Integration Tests
# =============================================================================

class TestAllSolutions:
    """Integration tests verifying all solutions work together."""

    @pytest.mark.parametrize("solution_module", [
        Collins_Stewart,
        Conformally_flat,
        EdS,
        Harvey_Tsoubelis,
        ICPertFLRW,
        LCDM,
        Non_diagonal,
        Rosquist_Jantzen,
        Schwarzschild_isotropic,
        Szekeres
    ])
    def test_solution_module_has_docstring(self, solution_module):
        """Test that each solution module has a docstring."""
        # Some modules may not have docstrings, which is acceptable
        if solution_module.__doc__ is not None:
            assert len(solution_module.__doc__) > 0

    @pytest.mark.parametrize("solution_module,expected_functions", [
        (Collins_Stewart, ['rho', 'press', 'gammadown3', 'Kdown3']),
        (Conformally_flat, ['alpha', 'gammadown3', 'Kdown3']),
        (EdS, ['a', 'Hprop', 'rho', 'alpha', 'betaup3']),
        (Harvey_Tsoubelis, ['rho', 'gammadown3', 'Kdown3']),
        (ICPertFLRW, ['gammadown3', 'Kdown3', 'delta1', 'Rc_func']),
        (LCDM, ['a', 'Hprop', 'rho', 'alpha']),
        (Non_diagonal, ['gammadown3', 'Kdown3', 'A', 'dzA', 'dzdzA', 'Tdown4', 'data']),
        (Rosquist_Jantzen, ['gammadown3', 'Kdown3']),  # No rho function
        (Schwarzschild_isotropic, ['alpha', 'betaup3', 'gammadown3', 'Kdown3',
                                   'Tdown4']),
        (Szekeres, ['rho', 'press'])
    ])
    def test_solution_has_expected_functions(self, solution_module, expected_functions):
        """Test that each solution has its expected key functions."""
        for func_name in expected_functions:
            assert hasattr(solution_module, func_name), \
                f"{solution_module.__name__} missing {func_name}"
            assert callable(getattr(solution_module, func_name)), \
                f"{func_name} in {solution_module.__name__} is not callable"
