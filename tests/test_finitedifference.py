"""Tests for the finitedifference module."""
import numpy as np
import pytest

from aurel import finitedifference


class TestFiniteDifferenceFunctions:
    """Test standalone finite difference functions."""

    @pytest.fixture
    def test_function_data(self):
        """Create test data using a simple polynomial f(x) = x^2."""
        dx = 0.1
        x = np.arange(-5, 5, dx)
        f = x**2
        analytical_derivative = 2 * x  # f'(x) = 2x
        return x, f, analytical_derivative, dx

    def test_fd2_centered(self, test_function_data):
        """Test 2nd order centered finite difference."""
        x, f, analytical, dx = test_function_data
        # Test at middle point
        i = len(f) // 2
        result = finitedifference.fd2_centered(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-2)

    def test_fd2_forward(self, test_function_data):
        """Test 2nd order forward finite difference."""
        x, f, analytical, dx = test_function_data
        # Test near the start
        i = 5
        result = finitedifference.fd2_forward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-2)

    def test_fd2_backward(self, test_function_data):
        """Test 2nd order backward finite difference."""
        x, f, analytical, dx = test_function_data
        # Test near the end
        i = len(f) - 5
        result = finitedifference.fd2_backward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-2)

    def test_fd4_centered(self, test_function_data):
        """Test 4th order centered finite difference."""
        x, f, analytical, dx = test_function_data
        i = len(f) // 2
        result = finitedifference.fd4_centered(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-4)

    def test_fd4_forward(self, test_function_data):
        """Test 4th order forward finite difference."""
        x, f, analytical, dx = test_function_data
        i = 5
        result = finitedifference.fd4_forward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-4)

    def test_fd4_backward(self, test_function_data):
        """Test 4th order backward finite difference."""
        x, f, analytical, dx = test_function_data
        i = len(f) - 5
        result = finitedifference.fd4_backward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-4)

    def test_fd6_centered(self, test_function_data):
        """Test 6th order centered finite difference."""
        x, f, analytical, dx = test_function_data
        i = len(f) // 2
        result = finitedifference.fd6_centered(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-6)

    def test_fd6_forward(self, test_function_data):
        """Test 6th order forward finite difference."""
        x, f, analytical, dx = test_function_data
        i = 10
        result = finitedifference.fd6_forward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-6)

    def test_fd6_backward(self, test_function_data):
        """Test 6th order backward finite difference."""
        x, f, analytical, dx = test_function_data
        i = len(f) - 10
        result = finitedifference.fd6_backward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-6)

    def test_fd8_centered(self, test_function_data):
        """Test 8th order centered finite difference."""
        x, f, analytical, dx = test_function_data
        i = len(f) // 2
        result = finitedifference.fd8_centered(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-8)

    def test_fd8_forward(self, test_function_data):
        """Test 8th order forward finite difference."""
        x, f, analytical, dx = test_function_data
        i = 10
        result = finitedifference.fd8_forward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-8)

    def test_fd8_backward(self, test_function_data):
        """Test 8th order backward finite difference."""
        x, f, analytical, dx = test_function_data
        i = len(f) - 10
        result = finitedifference.fd8_backward(f, i, 1/dx)
        expected = analytical[i]
        assert np.isclose(result, expected, rtol=1e-8)

    def test_fd_order_accuracy_comparison(self):
        """Test that higher order schemes are more accurate."""
        dx = 0.1
        x = np.arange(-5, 5, dx)
        # Use a function with more interesting derivatives
        f = np.sin(x) * np.exp(x/2)
        analytical = np.cos(x) * np.exp(x/2) + 0.5 * np.sin(x) * np.exp(x/2)

        i = len(f) // 2

        # Compute errors for different orders
        error2 = abs(finitedifference.fd2_centered(f, i, 1/dx) - analytical[i])
        error4 = abs(finitedifference.fd4_centered(f, i, 1/dx) - analytical[i])
        error6 = abs(finitedifference.fd6_centered(f, i, 1/dx) - analytical[i])
        error8 = abs(finitedifference.fd8_centered(f, i, 1/dx) - analytical[i])

        # Higher order should have smaller errors
        assert error4 < error2
        assert error6 < error4
        assert error8 < error6


class TestFiniteDifferenceClass:
    """Test the FiniteDifference class."""

    @pytest.fixture
    def params_3d(self):
        """Create 3D grid parameters for testing."""
        return {
            'xmin': -2.0, 'ymin': -2.0, 'zmin': -2.0,
            'dx': 0.1, 'dy': 0.1, 'dz': 0.1,
            'Nx': 41, 'Ny': 41, 'Nz': 41
        }

    @pytest.fixture
    def test_scalar_field(self, params_3d):
        """Create a test scalar field f(x,y,z) = x^2 + y^2 + z^2."""
        x = np.arange(params_3d['xmin'],
                     params_3d['xmin'] + params_3d['Nx'] * params_3d['dx'],
                     params_3d['dx'])
        y = np.arange(params_3d['ymin'],
                     params_3d['ymin'] + params_3d['Ny'] * params_3d['dy'],
                     params_3d['dy'])
        z = np.arange(params_3d['zmin'],
                     params_3d['zmin'] + params_3d['Nz'] * params_3d['dz'],
                     params_3d['dz'])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = X**2 + Y**2 + Z**2
        # Analytical derivatives: df/dx = 2x, df/dy = 2y, df/dz = 2z
        return f, X, Y, Z

    @pytest.fixture
    def test_vector_field(self, params_3d):
        """Create a test vector field V = [x^2, y^2, z^2]."""
        x = np.arange(params_3d['xmin'],
                     params_3d['xmin'] + params_3d['Nx'] * params_3d['dx'],
                     params_3d['dx'])
        y = np.arange(params_3d['ymin'],
                     params_3d['ymin'] + params_3d['Ny'] * params_3d['dy'],
                     params_3d['dy'])
        z = np.arange(params_3d['zmin'],
                     params_3d['zmin'] + params_3d['Nz'] * params_3d['dz'],
                     params_3d['dz'])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        V = np.array([X**2, Y**2, Z**2])
        return V, X, Y, Z

    def test_initialization_order2(self, params_3d):
        """Test FiniteDifference initialization with 2nd order."""
        fd = finitedifference.FiniteDifference(params_3d, fd_order=2, verbose=False)
        assert fd.fd_order == 2
        assert fd.mask_len == 1
        assert fd.Nx == params_3d['Nx']
        assert fd.Ny == params_3d['Ny']
        assert fd.Nz == params_3d['Nz']

    def test_initialization_order4(self, params_3d):
        """Test FiniteDifference initialization with 4th order."""
        fd = finitedifference.FiniteDifference(params_3d, fd_order=4, verbose=False)
        assert fd.fd_order == 4
        assert fd.mask_len == 2
        assert fd.backward == finitedifference.fd4_backward
        assert fd.centered == finitedifference.fd4_centered
        assert fd.forward == finitedifference.fd4_forward

    def test_initialization_order6(self, params_3d):
        """Test FiniteDifference initialization with 6th order."""
        fd = finitedifference.FiniteDifference(params_3d, fd_order=6, verbose=False)
        assert fd.fd_order == 6
        assert fd.mask_len == 3
        assert fd.backward == finitedifference.fd6_backward
        assert fd.centered == finitedifference.fd6_centered
        assert fd.forward == finitedifference.fd6_forward

    def test_initialization_order8(self, params_3d):
        """Test FiniteDifference initialization with 8th order."""
        fd = finitedifference.FiniteDifference(params_3d, fd_order=8, verbose=False)
        assert fd.fd_order == 8
        assert fd.mask_len == 4
        assert fd.backward == finitedifference.fd8_backward
        assert fd.centered == finitedifference.fd8_centered
        assert fd.forward == finitedifference.fd8_forward

    def test_d3x_no_boundary(self, params_3d, test_scalar_field):
        """Test d3x with no boundary conditions."""
        f, X, Y, Z = test_scalar_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3x(f)

        # Check shape is preserved
        assert result.shape == f.shape

        # Check analytical derivative in interior (away from boundaries)
        mask = fd.mask_len
        interior_result = result[mask:-mask, mask:-mask, mask:-mask]
        interior_analytical = 2 * X[mask:-mask, mask:-mask, mask:-mask]

        assert np.allclose(interior_result, interior_analytical, rtol=1e-4)

    def test_d3y_no_boundary(self, params_3d, test_scalar_field):
        """Test d3y with no boundary conditions."""
        f, X, Y, Z = test_scalar_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3y(f)

        # Check shape is preserved
        assert result.shape == f.shape

        # Check analytical derivative in interior
        mask = fd.mask_len
        interior_result = result[mask:-mask, mask:-mask, mask:-mask]
        interior_analytical = 2 * Y[mask:-mask, mask:-mask, mask:-mask]

        assert np.allclose(interior_result, interior_analytical, rtol=1e-4)

    def test_d3z_no_boundary(self, params_3d, test_scalar_field):
        """Test d3z with no boundary conditions."""
        f, X, Y, Z = test_scalar_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3z(f)

        # Check shape is preserved
        assert result.shape == f.shape

        # Check analytical derivative in interior
        mask = fd.mask_len
        interior_result = result[mask:-mask, mask:-mask, mask:-mask]
        interior_analytical = 2 * Z[mask:-mask, mask:-mask, mask:-mask]

        assert np.allclose(interior_result, interior_analytical, rtol=1e-4)

    def test_d3_periodic_boundary(self, params_3d):
        """Test d3 with periodic boundary conditions."""
        # Create a periodic function: sin(2*pi*x/L)
        x = np.arange(params_3d['xmin'],
                     params_3d['xmin'] + params_3d['Nx'] * params_3d['dx'],
                     params_3d['dx'])
        y = np.arange(params_3d['ymin'],
                     params_3d['ymin'] + params_3d['Ny'] * params_3d['dy'],
                     params_3d['dy'])
        z = np.arange(params_3d['zmin'],
                     params_3d['zmin'] + params_3d['Nz'] * params_3d['dz'],
                     params_3d['dz'])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        L = params_3d['Nx'] * params_3d['dx']
        f = np.sin(2 * np.pi * X / L)

        fd = finitedifference.FiniteDifference(params_3d, boundary='periodic',
                                               fd_order=4, verbose=False)
        result = fd.d3x(f)

        # Analytical derivative: d/dx[sin(2*pi*x/L)] = (2*pi/L)*cos(2*pi*x/L)
        analytical = (2 * np.pi / L) * np.cos(2 * np.pi * X / L)

        # With periodic boundaries, accuracy should be good throughout
        assert np.allclose(result, analytical, rtol=1e-3)

    def test_d3_symmetric_boundary(self, params_3d):
        """Test d3 with symmetric boundary conditions."""
        # Create a symmetric function: x^2
        x = np.arange(params_3d['xmin'],
                     params_3d['xmin'] + params_3d['Nx'] * params_3d['dx'],
                     params_3d['dx'])
        y = np.arange(params_3d['ymin'],
                     params_3d['ymin'] + params_3d['Ny'] * params_3d['dy'],
                     params_3d['dy'])
        z = np.arange(params_3d['zmin'],
                     params_3d['zmin'] + params_3d['Nz'] * params_3d['dz'],
                     params_3d['dz'])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        f = X**2

        fd = finitedifference.FiniteDifference(params_3d, boundary='symmetric',
                                               fd_order=4, verbose=False)
        result = fd.d3x(f)

        # Analytical derivative: d/dx[x^2] = 2x
        analytical = 2 * X

        # With symmetric boundaries, check interior accuracy
        mask = fd.mask_len
        assert np.allclose(result[mask:-mask, mask:-mask, mask:-mask],
                          analytical[mask:-mask, mask:-mask, mask:-mask], rtol=1e-3)

    def test_d3_scalar(self, params_3d, test_scalar_field):
        """Test d3_scalar for gradient computation."""
        f, X, Y, Z = test_scalar_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3_scalar(f)

        # Result should be a 3-component vector field
        assert result.shape == (3,) + f.shape

        # Check each component in interior
        mask = fd.mask_len
        assert np.allclose(result[0, mask:-mask, mask:-mask, mask:-mask],
                          2 * X[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)
        assert np.allclose(result[1, mask:-mask, mask:-mask, mask:-mask],
                          2 * Y[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)
        assert np.allclose(result[2, mask:-mask, mask:-mask, mask:-mask],
                          2 * Z[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)

    def test_d3x_rank1tensor(self, params_3d, test_vector_field):
        """Test d3x_rank1tensor for vector field derivatives."""
        V, X, Y, Z = test_vector_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3x_rank1tensor(V)

        # Result should have shape (3, Nx, Ny, Nz) for dV_i/dx
        assert result.shape == V.shape

        # Check analytical derivatives in interior
        # dV_x/dx = d(x^2)/dx = 2x
        # dV_y/dx = d(y^2)/dx = 0
        # dV_z/dx = d(z^2)/dx = 0
        mask = fd.mask_len
        assert np.allclose(result[0, mask:-mask, mask:-mask, mask:-mask],
                          2 * X[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)
        assert np.allclose(result[1, mask:-mask, mask:-mask, mask:-mask],
                          0, atol=1e-4)
        assert np.allclose(result[2, mask:-mask, mask:-mask, mask:-mask],
                          0, atol=1e-4)

    def test_d3y_rank1tensor(self, params_3d, test_vector_field):
        """Test d3y_rank1tensor for vector field derivatives."""
        V, X, Y, Z = test_vector_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3y_rank1tensor(V)

        # Check analytical derivatives in interior
        # dV_x/dy = d(x^2)/dy = 0
        # dV_y/dy = d(y^2)/dy = 2y
        # dV_z/dy = d(z^2)/dy = 0
        mask = fd.mask_len
        assert np.allclose(result[0, mask:-mask, mask:-mask, mask:-mask],
                          0, atol=1e-4)
        assert np.allclose(result[1, mask:-mask, mask:-mask, mask:-mask],
                          2 * Y[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)
        assert np.allclose(result[2, mask:-mask, mask:-mask, mask:-mask],
                          0, atol=1e-4)

    def test_d3z_rank1tensor(self, params_3d, test_vector_field):
        """Test d3z_rank1tensor for vector field derivatives."""
        V, X, Y, Z = test_vector_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3z_rank1tensor(V)

        # Check analytical derivatives in interior
        # dV_x/dz = d(x^2)/dz = 0
        # dV_y/dz = d(y^2)/dz = 0
        # dV_z/dz = d(z^2)/dz = 2z
        mask = fd.mask_len
        assert np.allclose(result[0, mask:-mask, mask:-mask, mask:-mask],
                          0, atol=1e-4)
        assert np.allclose(result[1, mask:-mask, mask:-mask, mask:-mask],
                          0, atol=1e-4)
        assert np.allclose(result[2, mask:-mask, mask:-mask, mask:-mask],
                          2 * Z[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)

    def test_d3_rank1tensor(self, params_3d, test_vector_field):
        """Test d3_rank1tensor for full gradient of vector field."""
        V, X, Y, Z = test_vector_field
        fd = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                               fd_order=4, verbose=False)
        result = fd.d3_rank1tensor(V)

        # Result should have shape (3, 3, Nx, Ny, Nz) for dV_j/dx_i
        assert result.shape == (3, 3, params_3d['Nx'], params_3d['Ny'], params_3d['Nz'])

        # Verify individual components
        mask = fd.mask_len
        # dV_x/dx = 2x
        assert np.allclose(result[0, 0, mask:-mask, mask:-mask, mask:-mask],
                          2 * X[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)
        # dV_y/dy = 2y
        assert np.allclose(result[1, 1, mask:-mask, mask:-mask, mask:-mask],
                          2 * Y[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)
        # dV_z/dz = 2z
        assert np.allclose(result[2, 2, mask:-mask, mask:-mask, mask:-mask],
                          2 * Z[mask:-mask, mask:-mask, mask:-mask], rtol=1e-4)

    def test_order_comparison_2_vs_4(self, params_3d):
        """Test that 4th order is more accurate than 2nd order."""
        # Use a more complex function to avoid hitting machine precision
        x = np.arange(params_3d['xmin'],
                     params_3d['xmin'] + params_3d['Nx'] * params_3d['dx'],
                     params_3d['dx'])
        y = np.arange(params_3d['ymin'],
                     params_3d['ymin'] + params_3d['Ny'] * params_3d['dy'],
                     params_3d['dy'])
        z = np.arange(params_3d['zmin'],
                     params_3d['zmin'] + params_3d['Nz'] * params_3d['dz'],
                     params_3d['dz'])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Use x^3 for more interesting higher derivatives
        f = X**3
        analytical = 3 * X**2

        fd2 = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                                fd_order=2, verbose=False)
        fd4 = finitedifference.FiniteDifference(params_3d, boundary='no boundary',
                                                fd_order=4, verbose=False)

        result2 = fd2.d3x(f)
        result4 = fd4.d3x(f)

        # Use larger mask for comparison
        mask = 4
        error2 = np.abs(result2[mask:-mask, mask:-mask, mask:-mask] -
                       analytical[mask:-mask, mask:-mask, mask:-mask]).max()
        error4 = np.abs(result4[mask:-mask, mask:-mask, mask:-mask] -
                       analytical[mask:-mask, mask:-mask, mask:-mask]).max()

        assert error4 < error2

    def test_cartesian_to_spherical_conversion(self, params_3d):
        """Test coordinate conversion from Cartesian to spherical."""
        fd = finitedifference.FiniteDifference(params_3d, verbose=False)

        # Test at a known point
        r, theta, phi = fd.cartesian_to_spherical(
            np.array([1.0]), np.array([0.0]), np.array([0.0]))

        assert np.isclose(r[0], 1.0)
        assert np.isclose(theta[0], np.pi/2)  # on equator
        assert np.isclose(phi[0], 0.0)

    def test_spherical_to_cartesian_conversion(self, params_3d):
        """Test coordinate conversion from spherical to Cartesian."""
        fd = finitedifference.FiniteDifference(params_3d, verbose=False)

        # Test at a known point
        x, y, z = fd.spherical_to_cartesian(
            np.array([1.0]), np.array([np.pi/2]), np.array([0.0]))

        assert np.isclose(x[0], 1.0)
        assert np.isclose(y[0], 0.0)
        assert np.isclose(z[0], 0.0)
