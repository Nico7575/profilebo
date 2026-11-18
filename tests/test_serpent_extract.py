import numpy as np
import pytest

from profilebo.evaluators.serpent.extract import parse_res_m

RES_M = """\
% Increase counter:
if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

ABS_KEFF                  (idx, [1:   2]) = [  1.20000E+00 0.00050 ];
ABS_KINF                  (idx, [1:   2]) = [  1.25000E+00 0.00040 ];
BURN_DAYS                 (idx, [1:   2]) = [  0.00000E+00 0.00000E+00 ];
BURNUP                    (idx, [1:   2]) = [  0.00000E+00 0.00000E+00 ];

ABS_KEFF                  (idx, [1:   2]) = [  1.10000E+00 0.00060 ];
ABS_KINF                  (idx, [1:   2]) = [  1.15000E+00 0.00045 ];
BURN_DAYS                 (idx, [1:   2]) = [  3.65000E+02 0.00000E+00 ];
BURNUP                    (idx, [1:   2]) = [  2.00000E+01 0.00000E+00 ];

ABS_KEFF                  (idx, [1:   2]) = [  1.00000E+00 0.00070 ] ;
ABS_KINF                  (idx, [1:   2]) = [  1.05000E+00 0.00050 ] ;
BURN_DAYS                 (idx, [1:   2]) = [  7.30000E+02 0.00000E+00 ] ;
BURNUP                    (idx, [1:   2]) = [  4.00000E+01 0.00000E+00 ] ;
"""


@pytest.fixture
def res_file(tmp_path):
    p = tmp_path / "case_res.m"
    p.write_text(RES_M)
    return p


def test_parse_series(res_file):
    traj = parse_res_m(res_file)
    np.testing.assert_allclose(traj.days, [0.0, 365.0, 730.0])
    np.testing.assert_allclose(traj.burnup, [0.0, 20.0, 40.0])
    np.testing.assert_allclose(traj.k_eff, [1.2, 1.1, 1.0])


def test_reactivity_conversion(res_file):
    traj = parse_res_m(res_file)
    # rho = (k-1)/k in pcm
    np.testing.assert_allclose(traj.rho[0], (1.2 - 1.0) / 1.2 * 1e5)
    np.testing.assert_allclose(traj.rho[2], 0.0, atol=1e-9)
    # sigma_rho = sigma_k / k^2 (delta method); sigma_k = k * rel
    np.testing.assert_allclose(traj.rho_std[0], 1.2 * 0.0005 / 1.2**2 * 1e5)


def test_missing_keff_raises(tmp_path):
    p = tmp_path / "bad_res.m"
    p.write_text("TOT_CPU_TIME (idx, [1: 1]) = [ 1.0 ];\n")
    with pytest.raises(ValueError, match="ABS_KEFF"):
        parse_res_m(p)


def test_a_transport_run_without_burnup_raises(tmp_path):
    p = tmp_path / "res.m"
    p.write_text("ABS_KEFF (idx, [1: 2]) = [ 1.1 0.001 ];\n")
    with pytest.raises(ValueError, match="BURN_DAYS"):
        parse_res_m(p)
