"""
(C) 2021 Genentech. All rights reserved.

Test file for t_opt package.

"""
import os
import sys
import uuid

import scripttest


TEST_DIR = "test_t_opt_outputs"


def run_script(*args, **kwargs):
    """"Runs Python script in test environment"""
    if not os.path.isdir(TEST_DIR):
        try:
            os.mkdir(TEST_DIR)
        except FileExistsError:
            pass
    _uuid = uuid.uuid4()
    testdir = os.path.join(TEST_DIR, _uuid.hex)
    env = scripttest.TestFileEnvironment(testdir)
    return env.run(sys.executable, *args, **kwargs)

def valid_oechem_license():
    """Returns true if there is no valid OEChem license."""
    # pylint: disable=import-outside-toplevel
    try:
        from openeye import oechem
    except ImportError:
        return False
    return oechem.OEChemIsLicensed()
