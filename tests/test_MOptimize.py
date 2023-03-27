import os
from tempfile import mkstemp
import difflib
import re
import logging
import logging.config as logConfig

from cdd_chem.util import io
from t_opt import sdf_multi_optimizer

logIni = os.path.dirname(__file__) + "/../t_opt/log.debug.ini"
logConfig.fileConfig(logIni, disable_existing_loggers=False)
log = logging.getLogger(__name__)

tmpDir=os.getenv("TMPDIR", os.getenv("TMP", "/tmp"))


def assert_sdf_equal(f1:str, f2:str):
    """ Diff two sd files and assert that they are equal
    """

    outStrg = io.read_file(f1)
    try:
        refStrg = io.read_file(f2)
    except FileNotFoundError as fnf:
        raise AssertionError(f"Reference File does not exist {f2} for {f1}") from fnf

    # drop mol timestamp line (2nd line in sdf file; not tested with rdkit and on windows)
    outStrg = re.sub('((^|\\$\\$\\$\\$[\r\n]).*[\r\n]).*([\r\n])','\\1\\3', outStrg).splitlines(1)
    refStrg = re.sub('((^|\\$\\$\\$\\$[\r\n]).*[\r\n]).*([\r\n])','\\1\\3', refStrg).splitlines(1)

    diff = ''.join(difflib.unified_diff(outStrg, refStrg))
    if diff:
        # pylint: disable=W1203
        log.info(f'sdf missmatch: {f1} {f2}')
        raise AssertionError(f'{diff} for  {f1}, {f2}')


def test_sdf_MOptimize():
    """ run sdf optimization with dummy potential and no constraints """

    args = " -in data/N2_1.0_0.9.sdf -out tmp -conf dummy"
    f, tmp = mkstemp(suffix='.sdf', dir=tmpDir)
    os.close(f)
    args = args.replace("tmp", tmp).split(" ")
    sdf_multi_optimizer.main(argv=args)

    # Diff to reference output
    assert_sdf_equal(tmp, 'data/test/N2_1.0_0.9.opt.ref.sdf')
    os.remove(tmp)


def test_sdf_MOptimize_harm():
    """ run sdf optimization with dummy potential and harmonic constraints """

    args = " -in data/N2_1.0_0.9.sdf -out tmp -conf dummy -harm_constr 0.2,0.4"
    f, tmp = mkstemp(suffix='.sdf', dir=tmpDir)
    os.close(f)
    args = args.replace("tmp", tmp).split(" ")
    sdf_multi_optimizer.main(argv=args)

    # Diff to reference output
    assert_sdf_equal(tmp, 'data/test/N2_1.0_0.9.harmopt.ref.sdf')
    os.remove(tmp)


def test_sdf_MOptimize_fix():
    """ run sdf optimization with dummy potential and fixed atom cosntraint """

    args = " -in data/C2H6.sdf -out tmp -conf dummy -constraint heavyAtom"
    f, tmp = mkstemp(suffix='.sdf', dir=tmpDir)
    os.close(f)
    args = args.replace("tmp", tmp).split(" ")
    sdf_multi_optimizer.main(argv=args)

    # Diff to reference output
    assert_sdf_equal(tmp, 'data/test/C2H6.fixOpt.ref.sdf')
    os.remove(tmp)
