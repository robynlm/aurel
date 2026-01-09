"""
Tests for the reading module.

These tests use real Einstein Toolkit simulation data stored in tests/fixtures/.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import aurel.reading as reading

# =============================================================================
#                               Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def simloc_env():
    """Set up SIMLOC environment variable pointing to fixtures.

    Cleans up generated cache files after tests complete.
    """
    # Save original SIMLOC if it exists
    original_simloc = os.environ.get('SIMLOC', None)

    # Set SIMLOC to fixtures path (fixtures/ is at same level as this test file)
    fixtures_path = Path(__file__).parent / "fixtures"
    os.environ['SIMLOC'] = str(fixtures_path) + '/'

    yield str(fixtures_path)

    # Cleanup: Remove generated cache files from fixtures
    import shutil
    for sim_dir in fixtures_path.glob("test_*/"):
        # Remove iterations.txt
        iterations_file = sim_dir / "iterations.txt"
        if iterations_file.exists():
            iterations_file.unlink()

        # Remove content.txt files in output directories
        for content_file in sim_dir.glob("output-*/*/content.txt"):
            content_file.unlink()

        # Remove all_iterations/ directories
        for all_it_dir in sim_dir.glob("output-*/*/all_iterations"):
            if all_it_dir.exists():
                shutil.rmtree(all_it_dir)

    # Restore original SIMLOC
    if original_simloc is not None:
        os.environ['SIMLOC'] = original_simloc
    else:
        del os.environ['SIMLOC']


# =============================================================================
#                         Test Parsing Functions
# =============================================================================

class TestParsing:
    """Tests for file and key parsing functions."""

    def test_parse_h5file_single_variable(self, simloc_env):
        """Test parsing single variable HDF5 filename."""
        fixtures_path = Path(simloc_env)
        filepath = (
            fixtures_path / "test_onefile_ungrouped/output-0000/"
            "test_onefile_ungrouped/alp.h5"
        )
        result = reading.parse_h5file(str(filepath))
        assert result is not None
        assert result['thorn_with_dash'] is None
        assert result['thorn'] is None
        assert result['variable_or_group'] == 'alp'
        assert result['base_name'] is None
        assert result['xyz_prefix'] is None
        assert result['chunk_number'] is None
        assert result['xyz_suffix'] is None
        assert result['group_file'] is False

    def test_parse_h5file_grouped_variable(self, simloc_env):
        """Test parsing grouped variable HDF5 filename."""
        fixtures_path = Path(simloc_env)
        filepath = (
            fixtures_path / "test_onefile_grouped/output-0000/"
            "test_onefile_grouped/admbase-shift.h5"
        )
        result = reading.parse_h5file(str(filepath))
        assert result is not None
        assert result['thorn_with_dash'] == 'admbase-'
        assert result['thorn'] == 'admbase'
        assert result['variable_or_group'] == 'shift'
        assert result['base_name'] == 'admbase-shift'
        assert result['xyz_prefix'] is None
        assert result['chunk_number'] is None
        assert result['xyz_suffix'] is None
        assert result['group_file'] is True

    def test_parse_h5file_with_chunks(self, simloc_env):
        """Test parsing chunked HDF5 filename."""
        fixtures_path = Path(simloc_env)
        filepath = (
            fixtures_path / "test_proc_ungrouped/output-0000/"
            "test_proc_ungrouped/betax.file_5.h5"
        )
        result = reading.parse_h5file(str(filepath))
        assert result is not None
        assert result['thorn_with_dash'] is None
        assert result['thorn'] is None
        assert result['variable_or_group'] == 'betax'
        assert result['base_name'] is None
        assert result['xyz_prefix'] is None
        assert result['chunk_number'] == 5
        assert result['xyz_suffix'] is None
        assert result['group_file'] is False

    def test_parse_h5file_grouped_chunked(self, simloc_env):
        """Test parsing grouped and chunked filename."""
        fixtures_path = Path(simloc_env)
        filepath = (
            fixtures_path / "test_proc_grouped/output-0000/"
            "test_proc_grouped/admbase-shift.file_7.h5"
        )
        result = reading.parse_h5file(str(filepath))
        assert result is not None
        assert result['thorn_with_dash'] == 'admbase-'
        assert result['thorn'] == 'admbase'
        assert result['variable_or_group'] == 'shift'
        assert result['base_name'] == 'admbase-shift'
        assert result['xyz_prefix'] is None
        assert result['chunk_number'] == 7
        assert result['xyz_suffix'] is None
        assert result['group_file'] is True

    def test_parse_h5file_checkpoint(self, simloc_env):
        """Test parsing checkpoint filename."""
        fixtures_path = Path(simloc_env)
        filepath = (
            fixtures_path / "test_onefile_ungrouped/output-0000/"
            "test_onefile_ungrouped/checkpoint.chkpt.it_0.h5"
        )
        result = reading.parse_h5file(str(filepath))
        assert result is not None
        assert result['iteration'] == 0
        assert result['chunk_number'] is None

        # Test with chunked checkpoint (hypothetical example)
        result_chunked = reading.parse_h5file("checkpoint.chkpt.it_100.file_3.h5")
        assert result_chunked is not None
        assert result_chunked['iteration'] == 100
        assert result_chunked['chunk_number'] == 3

    def test_parse_hdf5_key(self):
        """Test parsing HDF5 dataset key."""
        key = "ADMBASE::betax it=0 tl=0 m=0 rl=0 c=2"
        result = reading.parse_hdf5_key(key)

        assert result is not None
        assert result['thorn'] == 'ADMBASE'
        assert result['variable'] == 'betax'
        assert result['it'] == 0
        assert result['tl'] == 0
        assert result['m'] == 0
        assert result['rl'] == 0
        assert result['c'] == 2
        assert result['combined variable name'] == 'ADMBASE::betax'

    def test_parse_hdf5_key_no_chunk(self):
        """Test parsing HDF5 dataset key without chunk number."""
        key = "ADMBASE::alp it=100 tl=0 rl=1"
        result = reading.parse_hdf5_key(key)

        assert result is not None
        assert result['thorn'] == 'ADMBASE'
        assert result['variable'] == 'alp'
        assert result['it'] == 100
        assert result['tl'] == 0
        assert result['m'] is None
        assert result['rl'] == 1
        assert result['c'] is None
        assert result['combined variable name'] == 'ADMBASE::alp'

    def test_parse_h5file_invalid_filename(self):
        """Test parsing invalid HDF5 filename returns None."""
        # Test various invalid filenames that don't match the regex patterns
        assert reading.parse_h5file("invalid.txt") is None
        assert reading.parse_h5file("notahdf5file.dat") is None
        assert reading.parse_h5file("random_file") is None
        assert reading.parse_h5file("checkpoint.wrong.it_0.h5") is None
        assert reading.parse_h5file("") is None


# =============================================================================
#                      Test Variable Name Transformations
# =============================================================================

class TestVariableTransformations:
    """Tests for variable name transformation functions."""

    def test_transform_vars_tensor_to_scalar(self):
        """Test transforming tensor names to scalar components."""
        result = reading.transform_vars_tensor_to_scalar(['betaup3', 'alpha'])
        assert 'betax' in result
        assert 'betay' in result
        assert 'betaz' in result
        assert 'alpha' in result
        assert len(result) == 4  # 3 shift components + alpha

    def test_transform_vars_aurel_to_ET(self):
        """Test transforming Aurel variable names to ET names."""
        result = reading.transform_vars_aurel_to_ET(['betaup3', 'alpha'])
        assert 'betax' in result
        assert 'betay' in result
        assert 'betaz' in result
        assert 'alp' in result

    def test_transform_vars_ET_to_aurel(self):
        """Test transforming single ET variable name to Aurel."""
        assert reading.transform_vars_ET_to_aurel('alp') == 'alpha'
        assert reading.transform_vars_ET_to_aurel('betax') == 'betax'

    def test_transform_vars_ET_to_aurel_groups(self):
        """Test transforming ET variable list to Aurel groups."""
        et_vars = ['betax', 'betay', 'betaz', 'alp']
        result = reading.transform_vars_ET_to_aurel_groups(et_vars)
        assert 'betaup3' in result
        assert 'alpha' in result
        assert 'betax' not in result  # Should be grouped

# =============================================================================
#                      Test Parameters Function
# =============================================================================

class TestParameters:
    """Tests for reading simulation parameters."""

    @pytest.mark.parametrize("simname", [
        'test_onefile_ungrouped',
        'test_onefile_grouped',
        'test_proc_ungrouped',
        'test_proc_grouped'
    ])
    def test_parameters(self, simloc_env, simname):
        """Test reading parameters from all test simulations.

        Note: simloc_env fixture sets SIMLOC env var but we don't use its return value.
        """
        param = reading.parameters(simname)

        # Check essential keys are present
        assert param['simname'] == simname
        assert param['simulation'] == 'ET'
        assert 'simpath' in param
        assert 'datapath' in param
        assert 'list_of_thorns' in param

        # Check grid parameters required by FiniteDifference class
        for coord in ['x', 'y', 'z']:
            # Required for FD: xmin/ymin/zmin, dx/dy/dz, Nx/Ny/Nz
            assert coord + 'min' in param, f"{coord}min missing"
            assert 'd' + coord in param, f"d{coord} missing"
            assert 'N' + coord in param, f"N{coord} missing"

            # Additional derived parameters
            assert coord + 'max' in param, f"{coord}max missing"
            assert 'L' + coord in param, f"L{coord} missing"

            # Verify values are appropriate types
            assert isinstance(param[coord + 'min'], (int, float))
            assert isinstance(param['d' + coord], (int, float))
            assert isinstance(param['N' + coord], int)
            assert param['d' + coord] > 0, f"d{coord} must be positive"
            assert param['N' + coord] > 0, f"N{coord} must be positive"

        # Check refinement levels
        assert 'max_refinement_levels' in param

    def test_parameters_missing_simloc(self):
        """Test that missing SIMLOC environment variable raises error."""
        # Save and remove SIMLOC
        original_simloc = os.environ.get('SIMLOC', None)
        if 'SIMLOC' in os.environ:
            del os.environ['SIMLOC']

        try:
            with pytest.raises(
                ValueError, match="Could not find environment variable SIMLOC"
            ):
                reading.parameters('test_onefile_ungrouped')
        finally:
            # Restore SIMLOC
            if original_simloc is not None:
                os.environ['SIMLOC'] = original_simloc

    def test_parameters_with_multiple_colons_in_line(self, simloc_env, tmp_path):
        """Test parsing parameter lines with multiple '::' (lines 719-720)."""
        # Create a test parameter file with multiple :: in a line
        test_sim_dir = tmp_path / "test_multicolon"
        test_output_dir = test_sim_dir / "output-0000"
        test_output_dir.mkdir(parents=True)

        par_file = test_output_dir / "test_multicolon.par"
        par_content = """
# Test parameter file
ActiveThorns = "Carpet"

CartGrid3D::type = "coordbase"
CoordBase::xmin = -10.0
CoordBase::xmax = 10.0
CoordBase::ymin = -10.0
CoordBase::ymax = 10.0
CoordBase::zmin = -10.0
CoordBase::zmax = 10.0
CoordBase::dx = 0.5
CoordBase::dy = 0.5
CoordBase::dz = 0.5

# Line with multiple :: symbols (testing lines 719-720)
Some::Thorn::Parameter = "value"
Another::Complex::Thorn::Setting = 42
"""
        par_file.write_text(par_content)

        # Temporarily add test sim to SIMLOC
        original_simloc = os.environ.get('SIMLOC', '')
        os.environ['SIMLOC'] = str(tmp_path).replace('\\', '/') + '/'

        try:
            param = reading.parameters('test_multicolon')
            # Check that parameters with multiple :: are parsed correctly
            assert 'Thorn::Parameter' in param
            assert param['Thorn::Parameter'] == 'value'
            assert 'Complex::Thorn::Setting' in param
            assert param['Complex::Thorn::Setting'] == 42
        finally:
            os.environ['SIMLOC'] = original_simloc

    def test_parameters_with_multiple_equals_in_value(self, simloc_env, tmp_path):
        """Test parsing parameter values with multiple '=' (lines 727-728)."""
        # Create a test parameter file with multiple = in value
        test_sim_dir = tmp_path / "test_multiequals"
        test_output_dir = test_sim_dir / "output-0000"
        test_output_dir.mkdir(parents=True)

        par_file = test_output_dir / "test_multiequals.par"
        par_content = """
# Test parameter file
ActiveThorns = "Carpet"

CartGrid3D::type = "coordbase"
CoordBase::xmin = -10.0
CoordBase::xmax = 10.0
CoordBase::ymin = -10.0
CoordBase::ymax = 10.0
CoordBase::zmin = -10.0
CoordBase::zmax = 10.0
CoordBase::dx = 0.5
CoordBase::dy = 0.5
CoordBase::dz = 0.5

# Lines with multiple = symbols (testing lines 727-728)
Some::equation = "a=b+c"
Another::formula = "x=y=z"
Complex::expression = "result = input1 = input2"
"""
        par_file.write_text(par_content)

        # Temporarily add test sim to SIMLOC
        original_simloc = os.environ.get('SIMLOC', '')
        os.environ['SIMLOC'] = str(tmp_path).replace('\\', '/') + '/'

        try:
            param = reading.parameters('test_multiequals')
            # Check that values with multiple = are preserved correctly
            assert 'equation' in param
            assert param['equation'] == 'a=b+c'
            assert 'formula' in param
            assert param['formula'] == 'x=y=z'
            assert 'expression' in param
            assert param['expression'] == 'result = input1 = input2'
        finally:
            os.environ['SIMLOC'] = original_simloc

    def test_parameters_invalid_sim(self, simloc_env):
        """Test that invalid simulation name raises error."""
        with pytest.raises(ValueError, match="Could not find simulation"):
            reading.parameters('nonexistent_simulation')


# =============================================================================
#                      Test Content Analysis
# =============================================================================

class TestGetContent:
    """Tests for get_content function."""

    @pytest.mark.parametrize("simname,restart", [
        ('test_onefile_ungrouped', 0),
        ('test_onefile_ungrouped', 1),
        ('test_onefile_ungrouped', 2),
        ('test_onefile_grouped', 0),
        ('test_onefile_grouped', 1),
        ('test_onefile_grouped', 2),
        ('test_proc_ungrouped', 0),
        ('test_proc_ungrouped', 1),
        ('test_proc_ungrouped', 2),
        ('test_proc_grouped', 0),
        ('test_proc_grouped', 1),
        ('test_proc_grouped', 2),
    ])
    def test_get_content_all_simulations(self, simloc_env, simname, restart):
        """Test analyzing content for all test simulations and all restarts."""
        param = reading.parameters(simname)
        content = reading.get_content(param, restart=restart, verbose=False)

        # Basic structure checks
        assert len(content) > 0, f"No content found for {simname} restart {restart}"
        assert isinstance(content, dict), "Content should be a dictionary"

        # Check that we have some variables
        all_vars = []
        for var_tuple in content.keys():
            assert isinstance(var_tuple, tuple), "Keys should be tuples"
            all_vars.extend(var_tuple)
        assert len(all_vars) > 0, "Should have at least one variable"

        # Check that file paths are strings and exist
        for var_tuple, file_list in content.items():
            assert isinstance(file_list, list), "File lists should be lists"
            assert len(file_list) > 0, f"No files for {var_tuple}"
            for filepath in file_list:
                assert isinstance(filepath, str), "File paths should be strings"
                assert Path(filepath).exists(), f"File does not exist: {filepath}"
                # Verify file path points to correct restart
                assert f'output-{restart:04d}' in filepath, \
                    f"File {filepath} should be from restart {restart}"

        # Verify cache file is created in correct location
        cache_file = (Path(param['simpath']) / param['simname'] /
                      f'output-{restart:04d}' / param['simname'] / 'content.txt')
        assert cache_file.exists(), f"Cache file should exist for restart {restart}"

        # Check if simulation has grouped or chunked files
        if '_grouped' in simname:
            # Grouped simulations should have multi-variable tuples
            has_grouped = any(len(var_tuple) > 1 for var_tuple in content.keys())
            # Note: May be True or False depending on how variables are organized
            assert has_grouped, "Grouped simulations should have multi-variable tuples"

    def test_get_content_caching(self, simloc_env):
        """Test that content.txt cache file is created and used."""
        param = reading.parameters('test_onefile_ungrouped')
        cache_file = (
            Path(param['simpath']) / param['simname'] / 'output-0000'
            / param['simname'] / 'content.txt'
        )

        # Remove cache if it exists
        if cache_file.exists():
            cache_file.unlink()

        # First call - should create cache
        content1 = reading.get_content(param, restart=0, verbose=False)
        assert cache_file.exists(), "Cache file should be created"

        # Second call - should read from cache
        content2 = reading.get_content(param, restart=0, verbose=False)

        # Both should return same content
        assert content1.keys() == content2.keys()
        for key in content1.keys():
            assert content1[key] == content2[key]

    def test_get_content_overwrite(self, simloc_env):
        """Test that overwrite=True regenerates the cache."""
        param = reading.parameters('test_onefile_ungrouped')
        cache_file = (
            Path(param['simpath']) / param['simname'] / 'output-0000'
            / param['simname'] / 'content.txt'
        )

        # First call to ensure cache exists
        content1 = reading.get_content(param, restart=0, verbose=False)
        assert cache_file.exists()

        # Get modification time
        import time
        mtime1 = cache_file.stat().st_mtime
        time.sleep(0.1)  # Ensure different timestamp

        # Call with overwrite=True
        content2 = reading.get_content(param, restart=0, overwrite=True, verbose=False)

        # Cache should be regenerated (newer timestamp)
        mtime2 = cache_file.stat().st_mtime
        assert mtime2 > mtime1, "Cache file should be regenerated"

        # Content should still be the same
        assert content1.keys() == content2.keys()

    def test_get_content_multiple_calls_idempotent(self, simloc_env):
        """Test that multiple calls to get_content are idempotent."""
        param = reading.parameters('test_proc_grouped')

        # Call multiple times
        content1 = reading.get_content(param, restart=0, verbose=False)
        content2 = reading.get_content(param, restart=0, verbose=False)
        content3 = reading.get_content(param, restart=0, verbose=False)

        # All should return identical results
        assert content1.keys() == content2.keys() == content3.keys()
        for key in content1.keys():
            assert content1[key] == content2[key] == content3[key]


# =============================================================================
#                      Test Iterations Analysis
# =============================================================================

class TestIterations:
    """Tests for iteration cataloging functions."""

    @pytest.mark.slow
    def test_iterations_and_read_iterations(self, simloc_env):
        """Test that iterations function creates catalog and can be read back."""
        param = reading.parameters('test_onefile_ungrouped')

        # Create catalog with iterations()
        its_available = reading.iterations(
            param, skip_last=False, verbose=False)

        # Check basic structure
        assert len(its_available) > 0
        # Check that restarts are present
        assert 0 in its_available
        assert 1 in its_available
        assert 2 in its_available
        assert 'overall' in its_available

        # Check catalog file was created
        catalog_file = Path(param['simpath']) / param['simname'] / 'iterations.txt'
        assert catalog_file.exists()

        # Now read the catalog back with read_iterations()
        its_read = reading.read_iterations(param, verbose=False)

        # Should return identical structure
        # (except 'overall' which read_iterations doesn't parse)
        assert len(its_read) > 0

        # Compare restart keys (excluding 'overall' which only iterations() adds)
        restart_keys_available = {k for k in its_available.keys() if k != 'overall'}
        restart_keys_read = set(its_read.keys())
        assert restart_keys_read == restart_keys_available

        # Check structure of each restart
        for restart_num, restart_data in its_read.items():
            # Should have iteration information
            assert 'its available' in restart_data
            assert 'checkpoints' in restart_data

            # For test_onefile_ungrouped, checkpoints should exist
            if restart_num in [0, 1, 2]:
                assert isinstance(restart_data['checkpoints'], list)
                assert len(restart_data['checkpoints']) > 0, (
                    f"Expected checkpoints for restart {restart_num} "
                    "in test_onefile_ungrouped"
                )

            # Data should match what iterations() returned
            assert restart_data == its_available[restart_num]

    def test_iterations_skip_last_then_update(self, simloc_env):
        """Test iterations with skip_last=True, then update with skip_last=False."""
        param = reading.parameters('test_proc_ungrouped')

        # First run with skip_last=True (default behavior, skips active restart)
        its_available_skipped = reading.iterations(
            param, skip_last=True, verbose=False)

        restart_keys_skipped = [
            k for k in its_available_skipped.keys() if k != 'overall'
        ]
        num_restarts_skipped = len(restart_keys_skipped)

        # Check catalog file was created
        catalog_file = Path(param['simpath']) / param['simname'] / 'iterations.txt'
        assert catalog_file.exists()

        # Now update with skip_last=False (process all restarts including last)
        its_available_full = reading.iterations(
            param, skip_last=False, verbose=False)

        restart_keys_full = [k for k in its_available_full.keys() if k != 'overall']
        num_restarts_full = len(restart_keys_full)

        # Should have at least as many restarts after skip_last=False
        assert num_restarts_full > num_restarts_skipped
        # Should have 'overall' key in both
        assert 'overall' in its_available_skipped
        assert 'overall' in its_available_full

    def test_iterations_no_restarts_to_process(self, simloc_env, tmp_path):
        """Test iterations raises error when no restarts to process (lines 909-915)."""
        # Create a simulation directory with NO output directories at all
        test_sim_dir = tmp_path / "test_no_output"
        test_sim_dir.mkdir(parents=True)

        # Create a minimal parameter file (but not in output-XXXX since we
        # want to test no restarts)
        # We need to create it somewhere the parameters() function can find
        # it temporarily
        # Actually, parameters() expects the file in output-XXXX/simname.par
        # So we create one output dir with par file, but then simulate all
        # restarts being done
        test_output_dir = test_sim_dir / "output-0000"
        test_output_dir.mkdir(parents=True)

        par_file = test_output_dir / "test_no_output.par"
        par_content = """
ActiveThorns = "Carpet"
CoordBase::xmin = -10.0
CoordBase::xmax = 10.0
CoordBase::ymin = -10.0
CoordBase::ymax = 10.0
CoordBase::zmin = -10.0
CoordBase::zmax = 10.0
CoordBase::dx = 0.5
CoordBase::dy = 0.5
CoordBase::dz = 0.5
"""
        par_file.write_text(par_content)

        # Create iterations.txt file showing all restarts as done
        iterations_file = test_sim_dir / "iterations.txt"
        iterations_content = """==================
RESTART 0
==================
"""
        iterations_file.write_text(iterations_content)

        # Temporarily add test sim to SIMLOC
        original_simloc = os.environ.get('SIMLOC', '')
        os.environ['SIMLOC'] = str(tmp_path).replace('\\', '/') + '/'

        try:
            param = reading.parameters('test_no_output')

            # Try to run iterations with skip_last=True
            # This should raise ImportError because all restarts are done
            # and skip_last=True
            with pytest.raises(
                ImportError, match="Nothing to process.*skip_last=False"
            ):
                reading.iterations(param, skip_last=True, verbose=False)
        finally:
            os.environ['SIMLOC'] = original_simloc

    def test_iterations_empty_restart_directory(self, simloc_env, tmp_path):
        """Test iterations handles empty restart directories (lines 939-942)."""
        # Create a simulation directory with an empty output directory (no h5 files)
        test_sim_dir = tmp_path / "test_empty_restart"
        test_output_dir = test_sim_dir / "output-0000" / "test_empty_restart"
        test_output_dir.mkdir(parents=True)

        # Create a minimal parameter file
        par_file = test_sim_dir / "output-0000" / "test_empty_restart.par"
        par_content = """
ActiveThorns = "Carpet"
CoordBase::xmin = -10.0
CoordBase::xmax = 10.0
CoordBase::ymin = -10.0
CoordBase::ymax = 10.0
CoordBase::zmin = -10.0
CoordBase::zmax = 10.0
CoordBase::dx = 0.5
CoordBase::dy = 0.5
CoordBase::dz = 0.5
"""
        par_file.write_text(par_content)

        # Create a checkpoint file so iteration processing can continue
        # This tests that lines 939-942 handle empty 3D data but checkpoint exists
        checkpoint_file = test_output_dir / "checkpoint.chkpt.it_0.h5"
        checkpoint_file.touch()  # Empty file is fine for this test

        # Temporarily add test sim to SIMLOC
        original_simloc = os.environ.get('SIMLOC', '')
        os.environ['SIMLOC'] = str(tmp_path).replace('\\', '/') + '/'

        try:
            param = reading.parameters('test_empty_restart')

            # Run iterations - should handle empty directory gracefully
            # Lines 939-942 log message "Could not find 3D data" but don't raise error
            its_available = reading.iterations(param, skip_last=False, verbose=False)

            # Should have processed restart 0 even though no 3D data
            assert 0 in its_available
            # The restart should not have 'var available' key since no 3D data was found
            assert 'var available' not in its_available[0]
            # But should have checkpoint information
            assert 'checkpoints' in its_available[0]
            assert 0 in its_available[0]['checkpoints']
        finally:
            os.environ['SIMLOC'] = original_simloc


# =============================================================================
#                      Test Aurel Format I/O
# =============================================================================

class TestAurelFormat:
    """Tests for Aurel format reading and writing."""

    def test_save_and_read_aurel_data(self, tmp_path):
        """Test saving and reading data in Aurel format."""
        # Create test data
        param = {'datapath': str(tmp_path).replace('\\', '/') + '/'}
        data = {
            'it': np.array([0, 1]),
            't': [0.0, 0.1],
            'rho': [np.ones((8, 8, 8)), np.ones((8, 8, 8)) * 2],
            'alpha': [np.ones((8, 8, 8)) * 0.5, np.ones((8, 8, 8)) * 0.6]
        }

        # Save data
        reading.save_data(param, data, it=[0, 1], vars=['rho', 'alpha'])

        # Check files were created
        assert (tmp_path / 'it_0.hdf5').exists()
        assert (tmp_path / 'it_1.hdf5').exists()

        # Read data back
        data_read = reading.read_aurel_data(
            param, it=[0, 1], vars=['rho', 'alpha'], verbose=False)

        assert 'rho' in data_read
        assert 'alpha' in data_read
        assert len(data_read['rho']) == 2
        np.testing.assert_array_equal(data_read['rho'][0], data['rho'][0])
        np.testing.assert_array_equal(data_read['alpha'][1], data['alpha'][1])

    def test_read_aurel_data_missing_iteration(self, tmp_path):
        """Test reading when some iterations are missing."""
        param = {'datapath': str(tmp_path).replace('\\', '/') + '/'}
        data = {
            'it': np.array([0]),
            't': [0.0],
            'rho': [np.ones((8, 8, 8))]
        }

        reading.save_data(param, data, it=[0], vars=['rho'])

        # Try to read including missing iteration
        data_read = reading.read_aurel_data(
            param, it=[0, 1, 2], vars=['rho'], verbose=False)

        assert data_read['rho'][0] is not None
        assert data_read['rho'][1] is None
        assert data_read['rho'][2] is None

    def test_save_data_with_refinement_levels(self, tmp_path):
        """Test saving data with different refinement levels."""
        param = {'datapath': str(tmp_path).replace('\\', '/') + '/'}

        # rl=0
        data_rl0 = {
            'it': np.array([0]),
            't': [0.0],
            'rho': [np.ones((8, 8, 8))]
        }
        reading.save_data(param, data_rl0, it=[0], rl=0)

        # rl=1
        data_rl1 = {
            'it': np.array([0]),
            't': [0.0],
            'rho': [np.ones((16, 16, 16)) * 2]
        }
        reading.save_data(param, data_rl1, it=[0], rl=1)

        # Read both back, read all variable available
        read_rl0 = reading.read_aurel_data(param, it=[0], rl=0, verbose=False)
        read_rl1 = reading.read_aurel_data(param, it=[0], rl=1, verbose=False)

        assert read_rl0['rho'][0].shape == (8, 8, 8)
        assert read_rl1['rho'][0].shape == (16, 16, 16)
        assert np.mean(read_rl1['rho'][0]) == 2.0

    def test_save_data_without_trailing_slash(self, tmp_path):
        """Test saving data when datapath doesn't end with '/'."""
        # Use path without trailing slash to test lines 358-360 in reading.py
        param = {'datapath': str(tmp_path / 'data')}  # No trailing slash

        data = {
            'it': np.array([0, 1]),
            't': [0.0, 0.1],
            'rho': [np.ones((8, 8, 8)), np.ones((8, 8, 8)) * 2]
        }

        # Save data - this should handle the missing trailing slash
        reading.save_data(param, data, it=[0, 1], vars=['rho'])

        # Check files were created in correct location
        expected_dir = tmp_path / 'data'
        assert expected_dir.exists(), "Directory should be created"
        assert (expected_dir / 'it_0.hdf5').exists()
        assert (expected_dir / 'it_1.hdf5').exists()

        # Read back to verify
        param_read = {'datapath': str(expected_dir) + '/'}  # With slash for reading
        data_read = reading.read_aurel_data(
            param_read, it=[0, 1], vars=['rho'], verbose=False)

        assert 'rho' in data_read
        np.testing.assert_array_equal(data_read['rho'][0], data['rho'][0])


# =============================================================================
#                      Test ET Data Reading
# =============================================================================

class TestETDataReading:
    """Tests for reading Einstein Toolkit simulation data."""

    @pytest.mark.slow
    @pytest.mark.parametrize("simname,restart", [
        ('test_onefile_ungrouped', 0),
        ('test_onefile_ungrouped', 1),
        ('test_onefile_ungrouped', 2),
        ('test_onefile_grouped', 0),
        ('test_onefile_grouped', 1),
        ('test_onefile_grouped', 2),
        ('test_proc_ungrouped', 0),
        ('test_proc_ungrouped', 1),
        ('test_proc_ungrouped', 2),
        ('test_proc_grouped', 0),
        ('test_proc_grouped', 1),
        ('test_proc_grouped', 2),
    ])
    def test_read_ET_data_all_simulations(self, simloc_env, simname, restart):
        """Test reading ET data from all test simulations and all restarts."""
        param = reading.parameters(simname)

        # Get available iterations for this restart
        its_available = reading.iterations(param, skip_last=False, verbose=False)
        assert restart in its_available, f"Restart {restart} not found for {simname}"

        restart_its = its_available[restart]
        assert len(restart_its['its available']) > 0, (
            f"No iterations available for {simname} restart {restart}"
        )

        # Read first available iteration from specified restart
        itmin, itmax, dit = restart_its['rl = 0']
        it_to_read = np.arange(itmin, itmax, dit)
        data = reading.read_ET_data(
            param, it=it_to_read, restart=restart, split_per_it=False, verbose=False)

        assert 'it' in data
        assert len(data['it']) > 0
        assert data['it'][0] == it_to_read[0], (
            f"Expected iteration {it_to_read[0]}, got {data['it'][0]}"
        )

        # Should have some variables (more than just 'it' and 't')
        var_keys = [k for k in data.keys() if k not in ['it', 't']]
        assert len(var_keys) > 0, f"No variables found for {simname} restart {restart}"

    @pytest.mark.slow
    def test_read_ET_data_multiple_restarts(self, simloc_env):
        """Test reading data across multiple restarts."""
        param = reading.parameters('test_onefile_ungrouped')

        # Read from all restarts (auto-detect)
        data = reading.read_ET_data(
            param, it=[0], restart=-1, split_per_it=False, verbose=False)

        assert 'it' in data
        assert len(data['it']) > 0

    @pytest.mark.slow
    def test_read_ET_data_with_refinement_levels(self, simloc_env):
        """Test reading data from refined levels."""
        param = reading.parameters('test_onefile_ungrouped')

        # Read at rl=0
        data_rl0 = reading.read_ET_data(
            param, it=[0], restart=0, rl=0, split_per_it=False, verbose=False)

        # Read at rl=1
        data_rl1 = reading.read_ET_data(
            param, it=[0], restart=0, rl=1, split_per_it=False, verbose=False)

        # Both should have data
        assert 'it' in data_rl0
        assert 'it' in data_rl1

    @pytest.mark.slow
    def test_read_ET_data_specific_variables(self, simloc_env):
        """Test reading specific variables only."""
        param = reading.parameters('test_onefile_ungrouped')

        # Request specific variables in Aurel naming
        data = reading.read_ET_data(
            param, it=[0], vars=['betaup3'],
            restart=0, split_per_it=False, verbose=False)

        assert 'it' in data
        # Check that we got the requested variables (may be in ET or Aurel names)
        var_keys = [k for k in data.keys() if k not in ['it', 't']]
        assert len(var_keys) >= 1

        # Request specific variables in Aurel naming
        data = reading.read_ET_data(
            param, it=[0], vars=['alpha'],
            restart=0, split_per_it=False, verbose=False)

        assert 'it' in data
        # Check that we got the requested variables (may be in ET or Aurel names)
        var_keys = [k for k in data.keys() if k not in ['it', 't']]
        assert len(var_keys) >= 1

    @pytest.mark.slow
    def test_read_ET_data_with_split_per_it(self, simloc_env):
        """Test reading with split_per_it=True to test caching and hybrid reading."""
        param = reading.parameters('test_onefile_ungrouped')

        # Get available iterations
        its_available = reading.iterations(param, skip_last=False, verbose=False)
        restart = 0
        restart_its = its_available[restart]
        itmin, itmax, dit = restart_its['rl = 0']

        # Select all available iterations
        all_its = list(np.arange(itmin, itmax, dit))
        assert len(all_its) >= 3, "Need at least 3 iterations for this test"

        # First two calls: read all but the last iteration
        it_to_read = all_its[:-1]

        # First call with split_per_it=True - reads from ET files and saves to cache
        data1 = reading.read_ET_data(
            param, it=it_to_read, vars=['betaup3'],
            restart=restart, split_per_it=True, verbose=False)

        assert 'it' in data1
        assert len(data1['it']) == len(it_to_read)
        assert 'betax' in data1 or 'betaup3' in data1  # Should have shift component

        # Check that cache files were created in all_iterations directory
        all_it_dir = (Path(param['simpath']) / param['simname'] /
                      f'output-{restart:04d}' / param['simname'] / 'all_iterations')
        assert all_it_dir.exists(), "all_iterations directory should be created"

        # Check that at least one iteration file was created
        it_files = list(all_it_dir.glob('it_*.hdf5'))
        assert len(it_files) > 0, "Should have created per-iteration cache files"

        # Second call with split_per_it=True - should read from cached files
        data2 = reading.read_ET_data(
            param, it=it_to_read, vars=['betaup3'],
            restart=restart, split_per_it=True, verbose=False)

        assert 'it' in data2
        assert len(data2['it']) == len(it_to_read)

        # Data should be identical
        assert np.array_equal(data1['it'], data2['it'])
        for var in data1.keys():
            if var not in ['it', 't']:
                for idx in range(len(it_to_read)):
                    if data1[var][idx] is not None and data2[var][idx] is not None:
                        np.testing.assert_array_equal(
                            data1[var][idx], data2[var][idx],
                            err_msg=(
                                f"Data mismatch for {var} at iteration "
                                f"{it_to_read[idx]}"
                            )
                        )

        # Third call with all iterations including the last one - tests hybrid approach
        # (reading some from cache, some from ET files)
        data3 = reading.read_ET_data(
            param, it=all_its, vars=['betaup3'],
            restart=restart, split_per_it=True, verbose=False)

        assert 'it' in data3
        assert len(data3['it']) == len(all_its)

        # First iterations should match data1 (from cache)
        for var in data1.keys():
            if var not in ['it', 't']:
                for idx in range(len(it_to_read)):
                    if (data1[var][idx] is not None
                        and data3[var][idx] is not None):
                        np.testing.assert_array_equal(
                            data1[var][idx], data3[var][idx],
                            err_msg=f"Cached data mismatch for {var}")

        # Last iteration should have been newly read from ET files
        last_idx = len(all_its) - 1
        for var in data3.keys():
            if var not in ['it', 't']:
                assert data3[var][last_idx] is not None, \
                    f"Should have read {var} for new iteration"

    @pytest.mark.slow
    def test_read_ET_data_with_checkpoints(self, simloc_env):
        """Test reading checkpoint data."""
        param = reading.parameters('test_onefile_ungrouped')

        # Try to read with checkpoint usage
        data = reading.read_ET_data(
            param, it=[0], restart=0,
            usecheckpoints=True, verbose=False)
        assert 'it' in data

    @pytest.mark.slow
    def test_read_ET_data_across_restarts(self, simloc_env):
        """Test reading iterations across multiple restarts with data joining."""
        param = reading.parameters('test_onefile_ungrouped')

        # Get available iterations across all restarts
        its_available = reading.iterations(param, skip_last=False, verbose=False)

        # Use overall iterations which span across restarts
        assert 'overall' in its_available, "Should have overall iterations"
        assert 'rl = 0' in its_available['overall'], "Should have rl=0 in overall"

        # Collect iterations from overall (which spans multiple restarts)
        overall_rl0 = its_available['overall']['rl = 0']
        its_to_read = []
        for segment in overall_rl0:
            if len(segment)==3:
                itmin, itmax, dit = segment
                # Take a few iterations from each segment
                segment_its = list(np.arange(itmin, min(itmin + dit * 3, itmax), dit))
                its_to_read.extend(segment_its)
            else:
                # Single iteration segment
                its_to_read.append(segment[0])

        # Read data across restarts with restart=-1 (auto-detect)
        data = reading.read_ET_data(
            param, it=its_to_read, restart=-1,
            vars=['betaup3'], split_per_it=False, verbose=False)

        assert 'it' in data
        assert len(data['it']) == len(its_to_read), \
            f"Should have {len(its_to_read)} iterations, got {len(data['it'])}"

        # Verify iterations are in correct order
        for idx, expected_it in enumerate(its_to_read):
            assert data['it'][idx] == expected_it, (
                f"Iteration mismatch at index {idx}: expected {expected_it}, "
                f"got {data['it'][idx]}"
            )

        # Check that we got variables from all restarts
        var_keys = [k for k in data.keys() if k not in ['it', 't']]
        assert len(var_keys) > 0, "Should have variables from joined data"

        # Verify all data arrays have correct length
        for var in var_keys:
            assert len(data[var]) == len(its_to_read), \
                f"Variable {var} should have {len(its_to_read)} entries"

    @pytest.mark.slow
    def test_read_ET_checkpoints_across_restarts(self, simloc_env):
        """Test reading checkpoint data across multiple restarts."""
        param = reading.parameters('test_onefile_ungrouped')

        # Get available checkpoints for all restarts
        its_available = reading.iterations(param, skip_last=False, verbose=False)

        # Collect checkpoint iterations from all restarts
        checkpoint_its = []
        for restart in its_available.keys():
            if restart != 'overall' and isinstance(restart, int):
                if 'checkpoints' in its_available[restart]:
                    chkpts = its_available[restart]['checkpoints']
                    # Take first few checkpoints from each restart if available
                    checkpoint_its.extend(chkpts)

        # Read checkpoint data across restarts with usecheckpoints=True
        data = reading.read_ET_data(
            param, it=checkpoint_its, restart=-1,
            usecheckpoints=True, verbose=False)

        assert 'it' in data
        assert len(data['it']) == len(checkpoint_its), \
            f"Should have {len(checkpoint_its)} checkpoint iterations"

        # Verify checkpoint iterations are in correct order
        for idx, expected_it in enumerate(checkpoint_its):
            assert data['it'][idx] == expected_it, (
                f"Checkpoint iteration mismatch: expected {expected_it}, "
                f"got {data['it'][idx]}"
            )

        # Check that we got variables from checkpoints
        var_keys = [k for k in data.keys() if k not in ['it', 't']]
        assert len(var_keys) > 0, "Should have variables from checkpoint data"


# =============================================================================
#                      Test Main read_data Function
# =============================================================================

class TestReadData:
    """Tests for the main read_data function."""

    @pytest.mark.slow
    def test_read_data_ET_format(self, simloc_env):
        """Test read_data with ET format (has 'simulation' key)."""
        param = reading.parameters('test_onefile_ungrouped')

        data = reading.read_data(param, it=[0], restart=0, verbose=False)

        assert 'it' in data
        assert data['it'][0] == 0

    def test_read_data_aurel_format(self, tmp_path):
        """Test read_data with Aurel format (no 'simulation' key)."""
        # Create some test data first
        param = {'datapath': str(tmp_path).replace('\\', '/') + '/'}
        test_data = {
            'it': np.array([0, 1]),
            't': [0.0, 0.1],
            'rho': [np.ones((8, 8, 8)), np.ones((8, 8, 8)) * 2]
        }
        reading.save_data(param, test_data, it=[0, 1])

        # Now read it back using read_data
        data = reading.read_data(param, it=[0, 1], verbose=False)

        assert 'it' in data
        assert len(data['it']) == 2
        assert 'rho' in data

    def test_read_data_empty_iteration_list(self):
        """Test that empty iteration list raises error."""
        param = {'datapath': '/tmp/'}

        with pytest.raises(ValueError, match="it can not be an empty list"):
            reading.read_data(param, it=[])


# =============================================================================
#                      Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    @pytest.mark.slow
    def test_full_workflow_onefile_ungrouped(self, simloc_env, tmp_path):
        """Test complete workflow: read ET data, save Aurel format, read back."""
        # Read ET data
        param_et = reading.parameters('test_onefile_ungrouped')
        data_et = reading.read_ET_data(
            param_et, it=[0], restart=0, split_per_it=False, verbose=False)

        # Save in Aurel format
        param_aurel = {'datapath': str(tmp_path).replace('\\', '/') + '/'}
        vars_to_save = [k for k in data_et.keys() if k not in ['it']][:3]
        # Save a few vars
        reading.save_data(param_aurel, data_et, it=[0], vars=vars_to_save)

        # Read back from Aurel format
        data_aurel = reading.read_aurel_data(
            param_aurel, it=[0], vars=vars_to_save[1:], verbose=False)

        assert 'it' in data_aurel
        # Check that at least some variables match
        for var in vars_to_save[1:]:
            if var in data_et and var in data_aurel:
                if data_et[var][0] is not None and data_aurel[var][0] is not None:
                    np.testing.assert_array_equal(
                        data_et[var][0], data_aurel[var][0])

    @pytest.mark.slow
    @pytest.mark.parametrize("simname", [
        'test_onefile_ungrouped',
        'test_onefile_grouped',
        'test_proc_ungrouped',
        'test_proc_grouped'
    ])
    def test_all_simulation_configurations(self, simloc_env, simname):
        """Test reading from all four simulation configurations."""
        param = reading.parameters(simname)

        data = reading.read_ET_data(
            param, it=[0], restart=0,
            split_per_it=False, verbose=False)

        # Should successfully read data
        assert 'it' in data
        assert len(data['it']) > 0

        # Should have variables
        num_vars = len([k for k in data.keys() if k not in ['it', 't']])
        assert num_vars > 0, f"{simname} should have variables"
