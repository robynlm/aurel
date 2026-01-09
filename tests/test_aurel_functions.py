"""Tests for aurel core and symbolic functions."""

import numpy as np
import pytest
import sympy as sp

import aurel


class TestAurelCoreFunctions:
    """Test aurel functions execute."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixture with AurelCore instance."""
        Lx, Ly, Lz = 60.0, 40.0, 30.0
        Nx, Ny, Nz = 30, 20, 15
        param = { 'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
                'xmin': -Lx/2, 'ymin': -Ly/2, 'zmin': -Lz/2,
                'dx': Lx/Nx, 'dy': Ly/Ny, 'dz': Lz/Nz}
        fd = aurel.FiniteDifference(param)
        self.rel = aurel.AurelCore(fd, verbose=False)
        self.param = param
        self.fd = fd

    @pytest.mark.parametrize("key", aurel.descriptions)
    def test_executable(self, key):
        """Test all functions with default (Minkowski vacuum) assumptions."""
        output = self.rel[key]
        assert output is not None

    def test_with_vacuum_flag(self):
        """Test vacuum-specific branches in Constraints and dts_Gamma_bssnok."""
        rel_vac = aurel.AurelCore(self.fd, vacuum=True, verbose=False)

        # These functions have vacuum-specific branches
        ham = rel_vac["Hamiltonian"]
        assert ham is not None

        mom = rel_vac["Momentum_Escale"]
        assert mom is not None

        gamma = rel_vac["dts_Gamma_bssnok"]
        assert gamma is not None

    def test_with_matter(self):
        """Test matter-related quantities with non-zero matter fields."""
        rel_matter = aurel.AurelCore(self.fd, vacuum=False, verbose=False)

        # Set non-trivial matter fields
        rel_matter.data['rho0'] = np.ones(self.rel.data_shape)
        rel_matter.data['press'] = 0.1 * np.ones(self.rel.data_shape)
        rel_matter.data['eps'] = 0.05 * np.ones(self.rel.data_shape)
        rel_matter.data['w_lorentz'] = 1.1 * np.ones(self.rel.data_shape)

        # Test matter quantities
        rho = rel_matter["rho"]
        assert rho is not None
        assert np.any(rho > 0)

        enthalpy = rel_matter["enthalpy"]
        assert enthalpy is not None

        # Test Hamiltonian with matter (should include matter terms)
        ham_matter = rel_matter["Hamiltonian"]
        assert ham_matter is not None

    def test_with_custom_metric_components(self):
        """Test branches when gdown4 is provided directly."""
        rel = aurel.AurelCore(self.fd, verbose=False)

        # Provide gdown4 directly (triggers different branch in gdet)
        alpha = 1.2 * np.ones(self.rel.data_shape)
        gxx = 1.1 * np.ones(self.rel.data_shape)
        rel.data['gdown4'] = np.array([
            [
                -alpha**2, np.zeros_like(alpha),
                np.zeros_like(alpha), np.zeros_like(alpha)
            ],
            [np.zeros_like(alpha), gxx, np.zeros_like(alpha), np.zeros_like(alpha)],
            [np.zeros_like(alpha), np.zeros_like(alpha), gxx, np.zeros_like(alpha)],
            [np.zeros_like(alpha), np.zeros_like(alpha), np.zeros_like(alpha), gxx]
        ])

        # This should use the gdown4 branch
        gdet = rel["gdet"]
        assert gdet is not None

        gup = rel["gup4"]
        assert gup is not None

    def test_with_custom_gammadown3(self):
        """Test branches when gammadown3 is provided directly."""
        rel = aurel.AurelCore(self.fd, verbose=False)

        # Provide gammadown3 directly (triggers different branches)
        gxx = 1.1 * np.ones(self.rel.data_shape)
        rel.data['gammadown3'] = np.array([
            [gxx, np.zeros_like(gxx), np.zeros_like(gxx)],
            [np.zeros_like(gxx), gxx, np.zeros_like(gxx)],
            [np.zeros_like(gxx), np.zeros_like(gxx), gxx]
        ])

        # These should use the gammadown3 branch
        gamma_det = rel["gammadet"]
        assert gamma_det is not None

        gamma_up = rel["gammaup3"]
        assert gamma_up is not None

    def test_with_custom_Kdown3(self):
        """Test branches when Kdown3 is provided directly."""
        rel = aurel.AurelCore(self.fd, verbose=False)

        # Provide Kdown3 directly (triggers different branches)
        kxx = 0.1 * np.ones(self.rel.data_shape)
        rel.data['Kdown3'] = np.array([
            [kxx, np.zeros_like(kxx), np.zeros_like(kxx)],
            [np.zeros_like(kxx), kxx, np.zeros_like(kxx)],
            [np.zeros_like(kxx), np.zeros_like(kxx), kxx]
        ])

        # These should use the Kdown3 branch
        k_up = rel["Kup3"]
        assert k_up is not None

        k_trace = rel["Ktrace"]
        assert k_trace is not None

        a_down = rel["Adown3"]
        assert a_down is not None

    def test_with_shift_vector(self):
        """Test branches when betaup3 is provided."""
        rel = aurel.AurelCore(self.fd, verbose=False)

        # Provide betaup3 directly (triggers different branches in betax/y/z)
        beta = 0.1 * np.ones(self.rel.data_shape)
        rel.data['betaup3'] = np.array([beta, beta, beta])

        # These should use the betaup3 branch
        betax = rel["betax"]
        assert betax is not None
        assert np.any(betax != 0)

        betay = rel["betay"]
        assert betay is not None

        betaz = rel["betaz"]
        assert betaz is not None

        beta_down = rel["betadown3"]
        assert beta_down is not None

        beta_mag = rel["betamag"]
        assert beta_mag is not None

    def test_with_dtbetaup3(self):
        """Test branches when dtbetaup3 is provided."""
        rel = aurel.AurelCore(self.fd, verbose=False)

        # Provide dtbetaup3 directly
        dtbeta = 0.05 * np.ones(self.rel.data_shape)
        rel.data['dtbetaup3'] = np.array([dtbeta, dtbeta, dtbeta])

        # These should use the dtbetaup3 branch
        dtbetax = rel["dtbetax"]
        assert dtbetax is not None
        assert np.any(dtbetax != 0)

        dtbetay = rel["dtbetay"]
        assert dtbetay is not None

        dtbetaz = rel["dtbetaz"]
        assert dtbetaz is not None

    def test_Ttrace_with_Tdown4(self):
        """Test Ttrace with and without Tdown4 cached."""
        # Without Tdown4 - should use shortcut
        rel1 = aurel.AurelCore(self.fd, verbose=False)
        ttrace1 = rel1["Ttrace"]
        assert ttrace1 is not None

        # With Tdown4 cached - should use trace4
        rel2 = aurel.AurelCore(self.fd, verbose=False)
        rel2["Tdown4"]  # Cache this first
        ttrace2 = rel2["Ttrace"]
        assert ttrace2 is not None

    def test_Weyl_Psi_with_cached_data(self):
        """Test Weyl_Psi with and without Weyl_Psi4r/i cached."""
        # Without cached Weyl_Psi4r/i - should compute from tetrad
        rel1 = aurel.AurelCore(self.fd, verbose=False)
        psi1 = rel1["Weyl_Psi"]
        assert psi1 is not None
        assert len(psi1) == 5

        # With cached Weyl_Psi4r/i - should use cached values
        rel2 = aurel.AurelCore(self.fd, verbose=False)
        rel2.data['Weyl_Psi4r'] = np.zeros(self.rel.data_shape)
        rel2.data['Weyl_Psi4i'] = np.zeros(self.rel.data_shape)
        psi2 = rel2["Weyl_Psi"]
        assert psi2 is not None
        assert len(psi2) == 5

    def test_different_tetrad_choice(self):
        """Test tetrad-dependent calculations with different tetrad."""
        rel_qk = aurel.AurelCore(self.fd, tetrad="quasi-Kinnersley", verbose=False)
        rel_other = aurel.AurelCore(self.fd, tetrad="other", verbose=False)

        # Both should work but may give different results
        weyl_qk = rel_qk["st_Weyl_down4"]
        assert weyl_qk is not None

        weyl_other = rel_other["st_Weyl_down4"]
        assert weyl_other is not None

    def test_with_cosmological_constant(self):
        """Test with non-zero cosmological constant."""
        rel = aurel.AurelCore(self.fd, Lambda=0.1, verbose=False)

        # Hamiltonian includes Lambda term
        ham = rel["Hamiltonian"]
        assert ham is not None

class TestAurelCoreSymbolicFunctions:
    """Test aurel symbolic functions execute."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixture with AurelCoreSymbolic instances."""
        # 4D coordinates (spacetime)
        t, x, y, z = sp.symbols('t x y z')
        self.coords_4d = [t, x, y, z]
        self.rel_4d = aurel.AurelCoreSymbolic(self.coords_4d, verbose=False)

        # 3D coordinates (spatial)
        x3, y3, z3 = sp.symbols('x y z')
        self.coords_3d = [x3, y3, z3]
        self.rel_3d = aurel.AurelCoreSymbolic(self.coords_3d, verbose=False)

    @pytest.mark.parametrize("key", aurel.symbolic_descriptions)
    def test_executable_4d(self, key):
        """Test all symbolic functions with 4D coordinates."""
        output = self.rel_4d[key]
        assert output is not None

    @pytest.mark.parametrize("key", aurel.symbolic_descriptions)
    def test_executable_3d(self, key):
        """Test all symbolic functions with 3D coordinates."""
        output = self.rel_3d[key]
        assert output is not None

    def test_riemann_down_with_cached_uddd(self):
        """Test Riemann_down when Riemann_uddd is already cached."""
        # First compute Riemann_uddd to cache it
        self.rel_4d["Riemann_uddd"]
        # Then compute Riemann_down (should use cached path)
        output = self.rel_4d["Riemann_down"]
        assert output is not None
        assert output.shape == (4, 4, 4, 4)

    def test_riemann_down_without_cached_uddd(self):
        """Test Riemann_down when Riemann_uddd is not cached."""
        # Don't compute Riemann_uddd first
        output = self.rel_4d["Riemann_down"]
        assert output is not None
        assert output.shape == (4, 4, 4, 4)

    def test_ricci_down_with_cached_uddd(self):
        """Test Ricci_down when Riemann_uddd is already cached."""
        # First compute Riemann_uddd to cache it
        self.rel_4d["Riemann_uddd"]
        # Then compute Ricci_down (should use cached path)
        output = self.rel_4d["Ricci_down"]
        assert output is not None
        assert output.shape == (4, 4)

    def test_ricci_down_without_cached_uddd(self):
        """Test Ricci_down when Riemann_uddd is not cached."""
        # Don't compute Riemann_uddd first
        output = self.rel_4d["Ricci_down"]
        assert output is not None
        assert output.shape == (4, 4)

    def test_custom_metric(self):
        """Test with a custom metric (Schwarzschild)."""
        t, r, theta, phi = sp.symbols('t r theta phi', real=True, positive=True)
        M = sp.Symbol('M', real=True, positive=True)

        rel = aurel.AurelCoreSymbolic([t, r, theta, phi], verbose=False)

        # Define Schwarzschild metric
        rel.data['gdown'] = sp.Matrix([
            [-(1 - 2*M/r), 0, 0, 0],
            [0, 1/(1 - 2*M/r), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2]
        ])

        # Test that calculations work with custom metric
        gup = rel["gup"]
        assert gup is not None

        gamma = rel["Gamma_udd"]
        assert gamma is not None
        assert gamma.shape == (4, 4, 4)
