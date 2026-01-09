"""Tests for the aurel.time.over_time function with various configurations."""
import numpy as np
import pytest

import aurel
from aurel.solutions import EdS as sol


class TestOverTimeFunction:
    """Test over_time function with different variable and estimation configurations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data with LCDM solution."""
        # Define grid and finite difference class
        param = {
            'Nx': 16, 'Ny': 16, 'Nz': 16,
            'xmin': 0.0, 'ymin': 0.0, 'zmin': 0.0,
            'dx': 20.0, 'dy': 20.0, 'dz': 20.0
        }
        self.fd = aurel.FiniteDifference(param, verbose=False)
        x, y, z = self.fd.cartesian_coords

        # Generate minimal test data
        self.Nt = 3
        tarray = np.linspace(1, 5, self.Nt)
        self.data = {key: [] for key in ['t', 'gammadown3', 'Kdown3', 'rho']}
        for t in tarray:
            self.data['t'].append(t)
            self.data['gammadown3'].append(sol.gammadown3(t, x, y, z))
            self.data['Kdown3'].append(sol.Kdown3(t, x, y, z))
            self.data['rho'].append(
                sol.rho(t) * np.ones((self.fd.Nx, self.fd.Ny, self.fd.Nz))
            )

    def test_no_vars_no_estimates(self):
        """Test over_time with no variables or estimates requested."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[],
            estimates=[],
            verbose=False
        )
        # Should return original data unchanged
        assert set(result.keys()) == set(self.data.keys())
        assert len(result['t']) == self.Nt

    def test_builtin_var_no_estimates(self):
        """Test over_time with built-in variable from descriptions."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=[],
            verbose=False
        )
        # Should have original keys plus gammadet
        assert 'gammadet' in result.keys()
        assert len(result['gammadet']) == self.Nt
        assert result['gammadet'].shape == (self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz)

    def test_multiple_builtin_vars_no_estimates(self):
        """Test over_time with multiple built-in variables."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet', 'null_ray_exp_in'],
            estimates=[],
            verbose=False
        )
        assert 'gammadet' in result.keys()
        assert 'null_ray_exp_in' in result.keys()
        assert len(result['gammadet']) == self.Nt
        assert len(result['null_ray_exp_in']) == self.Nt

    def test_custom_var_no_estimates(self):
        """Test over_time with custom variable function."""
        def custom_var(rel):
            """Calculate custom variable: sum of gamma components."""
            return rel['gxx'] + rel['gyy'] + rel['gzz']

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'custom_trace': custom_var}],
            estimates=[],
            verbose=False
        )
        assert 'custom_trace' in result.keys()
        assert len(result['custom_trace']) == self.Nt
        assert result['custom_trace'].shape == (
            self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz
        )

    def test_mixed_vars(self):
        """Test over_time with both built-in and custom variables."""
        def custom_var(rel):
            """Calculate custom variable: sum of gamma components."""
            return rel['gxx'] + rel['gyy'] + rel['gzz']

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet', {'custom_trace': custom_var}],
            estimates=[],
            verbose=False
        )
        assert 'gammadet' in result.keys()
        assert 'custom_trace' in result.keys()
        assert len(result['gammadet']) == self.Nt
        assert len(result['custom_trace']) == self.Nt

    def test_builtin_var_builtin_estimate(self):
        """Test over_time with built-in variable and built-in estimation."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=['max'],
            verbose=False
        )
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert len(result['gammadet']) == self.Nt
        assert len(result['gammadet_max']) == self.Nt
        # Max should be a scalar for each timestep
        assert result['gammadet_max'].shape == (self.Nt,)

    def test_multiple_builtin_estimates(self):
        """Test over_time with multiple built-in estimations."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=['max', 'min', 'mean'],
            verbose=False
        )
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_min' in result.keys()
        assert 'gammadet_mean' in result.keys()
        for est in ['max', 'min', 'mean']:
            assert result[f'gammadet_{est}'].shape == (self.Nt,)

    def test_custom_estimate(self):
        """Test over_time with custom estimation function."""
        def custom_est(array):
            """Estimate custom value: value at center of grid."""
            Nx, Ny, Nz = np.shape(array)
            xcenter = Nx // 2
            ycenter = Ny // 2
            zcenter = Nz // 2
            return array[xcenter, ycenter, zcenter]

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=[{'center': custom_est}],
            verbose=False
        )
        assert 'gammadet_center' in result.keys()
        assert result['gammadet_center'].shape == (self.Nt,)

    def test_mixed_estimates(self):
        """Test over_time with both built-in and custom estimations."""
        def custom_est(array):
            return array[0, 0, 0]

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=['max', {'corner': custom_est}],
            verbose=False
        )
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_corner' in result.keys()
        assert result['gammadet_max'].shape == (self.Nt,)
        assert result['gammadet_corner'].shape == (self.Nt,)

    def test_estimates_on_input_scalars(self):
        """Test that estimates are also applied to input 3D scalar arrays."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[],
            estimates=['max', 'min'],
            verbose=False
        )
        # Should have estimates for input rho
        assert 'rho_max' in result.keys()
        assert 'rho_min' in result.keys()
        assert result['rho_max'].shape == (self.Nt,)

    def test_full_configuration(self):
        """Test over_time with all combinations: builtin+custom vars and estimates."""
        def custom_var(rel):
            return rel['gammadown3'][0, 0]

        def custom_est(array):
            return np.median(array)

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet', {'custom_var': custom_var}],
            estimates=['max', 'mean', {'custom_median': custom_est}],
            verbose=False
        )

        # Check variables
        assert 'gammadet' in result.keys()
        assert 'custom_var' in result.keys()

        # Check estimates on all 3D arrays (including input rho and computed vars)
        for var in ['gammadet', 'custom_var', 'rho']:
            assert f'{var}_max' in result.keys()
            assert f'{var}_mean' in result.keys()
            assert f'{var}_custom_median' in result.keys()
            assert result[f'{var}_max'].shape == (self.Nt,)

    def test_invalid_builtin_var(self):
        """Test that invalid built-in variable names are handled gracefully."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['invalid_variable_name'],
            estimates=[],
            verbose=False
        )
        # Should skip invalid variable and not crash
        assert 'invalid_variable_name' not in result.keys()

    def test_invalid_builtin_estimate(self):
        """Test that invalid built-in estimation names are handled gracefully."""
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=['invalid_estimate_name'],
            verbose=False
        )
        # Should skip invalid estimate and not crash
        assert 'gammadet_invalid_estimate_name' not in result.keys()

    def test_temporal_ordering(self):
        """Test that results maintain temporal ordering."""
        # Shuffle input data
        import random
        indices = list(range(self.Nt))
        random.shuffle(indices)

        shuffled_data = {key: [] for key in self.data.keys()}
        for idx in indices:
            for key in self.data.keys():
                shuffled_data[key].append(self.data[key][idx])

        result = aurel.over_time(
            shuffled_data,
            self.fd,
            vars=['gammadet'],
            estimates=['max'],
            verbose=False
        )

        # Results should be sorted by time
        assert np.all(np.diff(result['t']) > 0)

    def test_estimates_all_builtin(self):
        """Test over_time with all available built-in estimation functions."""
        builtin_estimates = list(aurel.time.est_functions.keys())

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=builtin_estimates,
            verbose=False
        )

        for est in builtin_estimates:
            assert f'gammadet_{est}' in result.keys()
            assert result[f'gammadet_{est}'].shape == (self.Nt,)

    def test_sequential_over_time_calls(self):
        """Test calling over_time sequentially to add variables with same estimates."""
        # First call: compute gammadet with max estimate
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=['max', 'mean'],
            verbose=False
        )

        # Verify first variable and estimates exist
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_mean' in result.keys()

        # Second call: add null_ray_exp_in with same estimates
        result = aurel.over_time(
            result,
            self.fd,
            vars=['gammadet', 'null_ray_exp_in'],
            verbose=False
        )

        # Verify both variables and their estimates exist
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_mean' in result.keys()
        assert 'null_ray_exp_in' in result.keys()

        # Third call: add null_ray_exp_in with same estimates
        result = aurel.over_time(
            result,
            self.fd,
            vars=['gammadet', 'null_ray_exp_in'],
            estimates=['max', 'mean'],
            verbose=False
        )

        # Verify both variables and their estimates exist
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_mean' in result.keys()
        assert 'null_ray_exp_in' in result.keys()
        assert 'null_ray_exp_in_max' in result.keys()
        assert 'null_ray_exp_in_mean' in result.keys()

        # Third call: add null_ray_exp_in with same estimates
        result = aurel.over_time(
            result,
            self.fd,
            vars=['gammadet', 'null_ray_exp_in', 'Ktrace'],
            estimates=['max', 'mean'],
            verbose=False
        )

        # Verify both variables and their estimates exist
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_mean' in result.keys()
        assert 'null_ray_exp_in' in result.keys()
        assert 'null_ray_exp_in_max' in result.keys()
        assert 'null_ray_exp_in_mean' in result.keys()
        assert 'Ktrace' in result.keys()
        assert 'Ktrace_max' in result.keys()
        assert 'Ktrace_mean' in result.keys()

        # Verify shapes
        assert result['gammadet'].shape == (self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz)
        assert result['gammadet_max'].shape == (self.Nt,)
        assert result['gammadet_mean'].shape == (self.Nt,)
        assert result['null_ray_exp_in'].shape == (
            self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz
        )
        assert result['null_ray_exp_in_max'].shape == (self.Nt,)
        assert result['null_ray_exp_in_mean'].shape == (self.Nt,)
        assert result['Ktrace'].shape == (self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz)
        assert result['Ktrace_max'].shape == (self.Nt,)
        assert result['Ktrace_mean'].shape == (self.Nt,)

    def test_sequential_vars_then_estimates_only(self):
        """Test calling over_time first with vars, then with only estimates."""
        # First call: compute variables without estimates
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet', 'Ktrace'],
            estimates=[],
            verbose=False
        )

        # Verify variables exist but no estimates
        assert 'gammadet' in result.keys()
        assert 'Ktrace' in result.keys()
        assert 'gammadet_max' not in result.keys()
        assert 'Ktrace_max' not in result.keys()

        # Second call: compute estimates only (no new vars)
        result = aurel.over_time(
            result,
            self.fd,
            vars=[],
            estimates=['max', 'min', 'mean'],
            verbose=False
        )

        # Verify variables still exist and now have estimates
        assert 'gammadet' in result.keys()
        assert 'Ktrace' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_min' in result.keys()
        assert 'gammadet_mean' in result.keys()
        assert 'Ktrace_max' in result.keys()
        assert 'Ktrace_min' in result.keys()
        assert 'Ktrace_mean' in result.keys()

        # Verify input scalar (rho) also got estimates
        assert 'rho_max' in result.keys()
        assert 'rho_min' in result.keys()
        assert 'rho_mean' in result.keys()

        # Verify shapes
        assert result['gammadet'].shape == (self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz)
        assert result['Ktrace'].shape == (self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz)
        assert result['gammadet_max'].shape == (self.Nt,)
        assert result['Ktrace_max'].shape == (self.Nt,)
        assert result['rho_max'].shape == (self.Nt,)

    def test_sequential_custom_vars_and_estimates(self):
        """Test sequential calls with custom variables and custom estimates."""
        def custom_var1(rel):
            """Calculate custom variable: trace of gamma."""
            return rel['gxx'] + rel['gyy'] + rel['gzz']

        def custom_var2(rel):
            """Calculate custom variable: squared rho."""
            return rel['rho'] ** 2

        def custom_est1(array):
            """Estimate custom value: center value."""
            Nx, Ny, Nz = np.shape(array)
            return array[Nx // 2, Ny // 2, Nz // 2]

        def custom_est2(array):
            """Estimate custom value: corner value."""
            return array[0, 0, 0]

        # First call: compute first custom var with builtin estimate
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet', {'gamma_trace': custom_var1}],
            estimates=['max'],
            verbose=False
        )

        # Verify first custom var and estimate exist
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gamma_trace' in result.keys()
        assert 'gamma_trace_max' in result.keys()

        # Second call: add second custom var with first custom estimate
        result = aurel.over_time(
            result,
            self.fd,
            vars=['Ktrace', {'rho_squared': custom_var2}],
            verbose=False
        )
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gamma_trace' in result.keys()
        assert 'gamma_trace_max' in result.keys()
        assert 'Ktrace' in result.keys()
        assert 'rho_squared' in result.keys()

        # Second call: add second custom var with first custom estimate
        result = aurel.over_time(
            result,
            self.fd,
            estimates=['max', {'center': custom_est1}],
            verbose=False
        )

        # Verify both custom vars exist with their estimates
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_center' in result.keys()
        assert 'gamma_trace' in result.keys()
        assert 'gamma_trace_max' in result.keys()
        assert 'gamma_trace_center' in result.keys()
        assert 'Ktrace' in result.keys()
        assert 'Ktrace_max' in result.keys()
        assert 'Ktrace_center' in result.keys()
        assert 'rho_squared' in result.keys()
        assert 'rho_squared_max' in result.keys()
        assert 'rho_squared_center' in result.keys()

        # Third call: add second custom estimate (no new vars)
        result = aurel.over_time(
            result,
            self.fd,
            vars=[],
            estimates=[{'corner': custom_est2}],
            verbose=False
        )

        # Verify all custom vars have all estimates
        assert 'gammadet' in result.keys()
        assert 'gammadet_max' in result.keys()
        assert 'gammadet_center' in result.keys()
        assert 'gamma_trace' in result.keys()
        assert 'gamma_trace_max' in result.keys()
        assert 'gamma_trace_center' in result.keys()
        assert 'Ktrace' in result.keys()
        assert 'Ktrace_max' in result.keys()
        assert 'Ktrace_center' in result.keys()
        assert 'rho_squared' in result.keys()
        assert 'rho_squared_max' in result.keys()
        assert 'rho_squared_center' in result.keys()
        assert 'gammadet_corner' in result.keys()
        assert 'gamma_trace_corner' in result.keys()
        assert 'Ktrace_corner' in result.keys()
        assert 'rho_squared_corner' in result.keys()

        # Verify input scalar also got custom estimates
        assert 'rho_center' in result.keys()
        assert 'rho_corner' in result.keys()

        # Verify shapes
        assert result['gamma_trace'].shape == (
            self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz
        )
        assert result['rho_squared'].shape == (
            self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz
        )
        assert result['gamma_trace_max'].shape == (self.Nt,)
        assert result['gamma_trace_center'].shape == (self.Nt,)
        assert result['gamma_trace_corner'].shape == (self.Nt,)
        assert result['rho_squared_center'].shape == (self.Nt,)
        assert result['rho_corner'].shape == (self.Nt,)

    def test_sequential_custom_vars_with_kwargs(self):
        """Test sequential calls with custom variables that use kwargs."""
        def custom_var_power(rel):
            """Calculate custom variable: rho raised to power."""
            return rel['rho'] ** rel.power

        def custom_var_scaled(rel):
            """Calculate custom variable: scaled gamma trace."""
            return rel.scale * (rel['gxx'] + rel['gyy'] + rel['gzz'])

        # First call: custom var with power=2
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'rho_squared': custom_var_power}],
            estimates=['max', 'min'],
            power=2,
            verbose=False
        )

        # Verify first custom var exists
        assert 'rho_squared' in result.keys()
        assert 'rho_squared_max' in result.keys()
        expected = self.data['rho'][0] ** 2
        np.testing.assert_array_almost_equal(result['rho_squared'][0], expected)

        # Second call: add another custom var with different kwarg
        result = aurel.over_time(
            result,
            self.fd,
            vars=[{'scaled_trace': custom_var_scaled}],
            estimates=['mean'],
            scale=3.5,
            verbose=False
        )

        # Verify both custom vars exist with their estimates
        assert 'rho_squared' in result.keys()
        assert 'rho_squared_max' in result.keys()
        assert 'rho_squared_min' in result.keys()
        assert 'scaled_trace' in result.keys()
        assert 'scaled_trace_mean' in result.keys()

        # Third call: recompute with different power
        result = aurel.over_time(
            result,
            self.fd,
            vars=[{'rho_cubed': custom_var_power}],
            estimates=[],
            power=3,
            verbose=False
        )

        # Verify new variable with different kwarg value
        assert 'rho_cubed' in result.keys()
        expected = self.data['rho'][0] ** 3
        np.testing.assert_array_almost_equal(result['rho_cubed'][0], expected)

        # Verify previous vars still exist
        assert 'rho_squared' in result.keys()
        assert 'scaled_trace' in result.keys()


class TestOverTimeEdgeCases:
    """Test edge cases and error handling for over_time function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up minimal test data."""
        param = {
            'Nx': 8, 'Ny': 8, 'Nz': 8,
            'xmin': 0.0, 'ymin': 0.0, 'zmin': 0.0,
            'dx': 10.0, 'dy': 10.0, 'dz': 10.0
        }
        self.fd = aurel.FiniteDifference(param, verbose=False)
        x, y, z = self.fd.cartesian_coords

        t = 1.0
        self.single_data = {
            't': [t],
            'gammadown3': [sol.gammadown3(t, x, y, z)],
            'Kdown3': [sol.Kdown3(t, x, y, z)],
            'rho': [sol.rho(t) * np.ones((self.fd.Nx, self.fd.Ny, self.fd.Nz))]
        }

    def test_single_timestep(self):
        """Test over_time with only one timestep."""
        result = aurel.over_time(
            self.single_data.copy(),
            self.fd,
            vars=['gammadet'],
            estimates=['max'],
            verbose=False
        )
        assert len(result['gammadet']) == 1
        assert len(result['gammadet_max']) == 1

    def test_missing_temporal_key(self):
        """Test that missing temporal key raises appropriate error."""
        bad_data = {
            'gammadown3': [sol.gammadown3(1.0, *self.fd.cartesian_coords)],
            'Kdown3': [sol.Kdown3(1.0, *self.fd.cartesian_coords)]
        }

        with pytest.raises(ValueError, match="temporal key"):
            aurel.over_time(
                bad_data,
                self.fd,
                vars=['gammadet'],
                verbose=False
            )

    def test_alternative_temporal_keys(self):
        """Test that alternative temporal keys work (it, iteration, time)."""
        for temp_key in ['it', 'iteration', 'time']:
            data = self.single_data.copy()
            data[temp_key] = data.pop('t')

            result = aurel.over_time(
                data,
                self.fd,
                vars=['gammadet'],
                estimates=[],
                verbose=False
            )
            assert temp_key in result.keys()
            assert 'gammadet' in result.keys()


class TestOverTimeCustomVarWithKwargs:
    """Test over_time with custom variable functions that accept kwargs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        param = {
            'Nx': 16, 'Ny': 16, 'Nz': 16,
            'xmin': 0.0, 'ymin': 0.0, 'zmin': 0.0,
            'dx': 20.0, 'dy': 20.0, 'dz': 20.0
        }
        self.fd = aurel.FiniteDifference(param, verbose=False)
        x, y, z = self.fd.cartesian_coords

        self.Nt = 3
        tarray = np.linspace(1, 5, self.Nt)
        self.data = {key: [] for key in ['t', 'gammadown3', 'Kdown3', 'rho']}
        for t in tarray:
            self.data['t'].append(t)
            self.data['gammadown3'].append(sol.gammadown3(t, x, y, z))
            self.data['Kdown3'].append(sol.Kdown3(t, x, y, z))
            self.data['rho'].append(
                sol.rho(t) * np.ones((self.fd.Nx, self.fd.Ny, self.fd.Nz))
            )

    def test_custom_var_with_int_kwarg(self):
        """Test custom variable with integer kwarg."""
        def custom_var(rel):
            """Calculate custom variable: rho raised to a power."""
            return rel['rho'] ** rel.power

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'rho_cubed': custom_var}],
            estimates=[],
            power=3,
            verbose=False
        )
        assert 'rho_cubed' in result.keys()
        expected = self.data['rho'][0] ** 3
        np.testing.assert_array_almost_equal(result['rho_cubed'][0], expected)

    def test_custom_var_with_float_kwarg(self):
        """Test custom variable with float kwarg."""
        def custom_var(rel):
            """Calculate custom variable: scaled rho."""
            return rel['rho'] * rel.scale

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'scaled_rho': custom_var}],
            estimates=[],
            scale=2.5,
            verbose=False
        )
        assert 'scaled_rho' in result.keys()
        expected = self.data['rho'][0] * 2.5
        np.testing.assert_array_almost_equal(result['scaled_rho'][0], expected)

    def test_custom_var_with_bool_kwarg(self):
        """Test custom variable with boolean kwarg."""
        def custom_var(rel):
            """Calculate custom variable: optionally take absolute value."""
            result = rel['gxx'] - rel['gyy']
            return np.abs(result) if rel.use_absolute else result

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'diff': custom_var}],
            estimates=[],
            use_absolute=True,
            verbose=False
        )
        assert 'diff' in result.keys()
        expected = np.abs(
            sol.gammadown3(self.data['t'][0], *self.fd.cartesian_coords)[0, 0]
            - sol.gammadown3(self.data['t'][0], *self.fd.cartesian_coords)[1, 1]
        )
        np.testing.assert_array_almost_equal(result['diff'][0], expected)

    def test_custom_var_with_string_kwarg(self):
        """Test custom variable with string kwarg."""
        def custom_var(rel):
            """Calculate custom variable: select gamma component by string."""
            return rel[f'g{rel.component}']

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'selected': custom_var}],
            estimates=[],
            component='yy',
            verbose=False
        )
        assert 'selected' in result.keys()
        expected = sol.gammadown3(self.data['t'][0], *self.fd.cartesian_coords)[1, 1]
        np.testing.assert_array_almost_equal(result['selected'][0], expected)

    def test_custom_var_with_list_kwarg(self):
        """Test custom variable with list kwarg."""
        def custom_var(rel):
            """Calculate custom variable: weighted sum of gamma diagonal."""
            return (rel.coefficients[0] * rel['gxx'] +
                    rel.coefficients[1] * rel['gyy'] +
                    rel.coefficients[2] * rel['gzz'])

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'weighted_trace': custom_var}],
            estimates=[],
            coefficients=[2, 3, 1],
            verbose=False
        )
        assert 'weighted_trace' in result.keys()
        gamma = sol.gammadown3(self.data['t'][0], *self.fd.cartesian_coords)
        expected = 2 * gamma[0, 0] + 3 * gamma[1, 1] + 1 * gamma[2, 2]
        np.testing.assert_array_almost_equal(result['weighted_trace'][0], expected)

    def test_custom_var_with_dict_kwarg(self):
        """Test custom variable with dict kwarg."""
        def custom_var(rel):
            """Calculate custom variable: affine transformation of rho."""
            return rel.config['scale'] * rel['rho'] + rel.config['offset']

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'transformed_rho': custom_var}],
            estimates=[],
            config={'scale': 3, 'offset': 5},
            verbose=False
        )
        assert 'transformed_rho' in result.keys()
        expected = 3 * self.data['rho'][0] + 5
        np.testing.assert_array_almost_equal(result['transformed_rho'][0], expected)

    def test_custom_var_with_set_kwarg(self):
        """Test custom variable with set kwarg."""
        def custom_var(rel):
            """Calculate custom variable: sum only specified components."""
            result = np.zeros_like(rel['gxx'])
            if 'xx' in rel.included_components:
                result += rel['gxx']
            if 'yy' in rel.included_components:
                result += rel['gyy']
            if 'zz' in rel.included_components:
                result += rel['gzz']
            return result

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'partial_trace': custom_var}],
            estimates=[],
            included_components={'xx', 'zz'},
            verbose=False
        )
        assert 'partial_trace' in result.keys()
        gamma = sol.gammadown3(self.data['t'][0], *self.fd.cartesian_coords)
        expected = gamma[0, 0] + gamma[2, 2]
        np.testing.assert_array_almost_equal(result['partial_trace'][0], expected)

    def test_custom_var_with_ndarray_kwarg(self):
        """Test custom variable with numpy array kwarg."""
        def custom_var(rel):
            """Calculate custom variable: spatially weighted sum."""
            return rel['rho'] * rel.weights

        weights = np.random.rand(self.fd.Nx, self.fd.Ny, self.fd.Nz)
        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'weighted_rho': custom_var}],
            estimates=[],
            weights=weights,
            verbose=False
        )
        assert 'weighted_rho' in result.keys()
        expected = self.data['rho'][0] * weights
        np.testing.assert_array_almost_equal(result['weighted_rho'][0], expected)

    def test_custom_var_with_function_kwarg(self):
        """Test custom variable with function kwarg."""
        def custom_var(rel):
            """Calculate custom variable: apply operation to rho."""
            return rel.operation(rel['rho'])

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'transformed': custom_var}],
            estimates=[],
            operation=np.square,
            verbose=False
        )
        assert 'transformed' in result.keys()
        expected = np.square(self.data['rho'][0])
        np.testing.assert_array_almost_equal(result['transformed'][0], expected)

    def test_custom_var_with_multiple_kwargs(self):
        """Test custom variable with multiple kwargs of different types."""
        def custom_var(rel):
            """Calculate custom variable."""
            result = np.zeros_like(rel['gxx'])
            for comp in rel.components:
                result += rel[f'g{comp}']
            result = result ** rel.power
            result = result * rel.scale
            if rel.use_log:
                result = np.log(np.abs(result) + 1e-10)
            return result

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[{'complex_var': custom_var}],
            estimates=[],
            scale=2.5,
            power=2,
            use_log=True,
            components=['xx', 'yy', 'zz'],
            verbose=False
        )
        assert 'complex_var' in result.keys()
        gamma = sol.gammadown3(self.data['t'][0], *self.fd.cartesian_coords)
        expected = gamma[0, 0] + gamma[1, 1] + gamma[2, 2]
        expected = expected ** 2
        expected = expected * 2.5
        expected = np.log(np.abs(expected) + 1e-10)
        np.testing.assert_array_almost_equal(result['complex_var'][0], expected)

    def test_multiple_custom_vars_with_different_kwargs(self):
        """Test multiple custom variables each using different kwargs."""
        def var1(rel):
            return rel['rho'] * rel.multiplier

        def var2(rel):
            return rel['rho'] ** rel.exponent

        result = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=[
                {'scaled': var1},
                {'powered': var2}
            ],
            estimates=[],
            multiplier=3,
            exponent=2,
            verbose=False
        )
        assert 'scaled' in result.keys()
        assert 'powered' in result.keys()
        np.testing.assert_array_almost_equal(
            result['scaled'][0], self.data['rho'][0] * 3
        )
        np.testing.assert_array_almost_equal(
            result['powered'][0], self.data['rho'][0] ** 2
        )

    def test_mutable_default_arguments(self):
        """Test that default arguments don't accumulate between calls.

        This test verifies that the fix for B006 (mutable default arguments)
        works correctly - each call to over_time should get fresh empty lists.
        """
        # First call with no vars or estimates (should use defaults)
        result1 = aurel.over_time(
            self.data.copy(),
            self.fd,
            verbose=False
        )

        # Second call with no vars or estimates (should also use defaults)
        result2 = aurel.over_time(
            self.data.copy(),
            self.fd,
            verbose=False
        )

        # Both should have the same keys (no accumulated state)
        assert set(result1.keys()) == set(self.data.keys())
        assert set(result2.keys()) == set(self.data.keys())

        # Third call with a var specified
        result3 = aurel.over_time(
            self.data.copy(),
            self.fd,
            vars=['gammadet'],
            verbose=False
        )

        # Fourth call with no vars should not have gammadet
        result4 = aurel.over_time(
            self.data.copy(),
            self.fd,
            verbose=False
        )

        assert 'gammadet' in result3.keys()
        assert 'gammadet' not in result4.keys()
