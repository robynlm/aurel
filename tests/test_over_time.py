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
            self.data['rho'].append(sol.rho(t) * np.ones((self.fd.Nx, self.fd.Ny, self.fd.Nz)))
    
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
            """Custom variable: sum of gamma components."""
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
        assert result['custom_trace'].shape == (self.Nt, self.fd.Nx, self.fd.Ny, self.fd.Nz)
    
    def test_mixed_vars(self):
        """Test over_time with both built-in and custom variables."""
        def custom_var(rel):
            """Custom variable: sum of gamma components."""
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
            """Custom estimation: value at center of grid."""
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
