"""Step 1 checks of the gravity-mass model at small n (n = 200)."""

import networkx as nx
import numpy as np
import pytest

from lab.gravity_mass import model

N = 200
TOL = 1e-10


@pytest.fixture(scope="module", params=["BA", "ER"])
def A(request):
    G = (
        nx.barabasi_albert_graph(N, 3, seed=1)
        if request.param == "BA"
        else nx.gnm_random_graph(N, 3 * N, seed=1)
    )
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.to_numpy_array(G)


@pytest.mark.parametrize("lam", [0.0, 1.0, 3.0])
def test_closed_form_stationary_matches_eigenvector(A, lam):
    g = model.standardize(np.random.default_rng(0).standard_normal(len(A)))
    P = model.kernel(A, g, lam)
    err = np.abs(model.stationary_closed_form(A, g, lam) - model.stationary_numeric(P)).max()
    assert err < TOL


def test_merw_is_log_psi_at_lambda_one(A):
    vals, vecs = np.linalg.eigh(A)
    psi = np.abs(vecs[:, -1])
    P_merw = A * psi[None, :] / (vals[-1] * psi[:, None])
    assert np.abs(model.kernel(A, np.log(psi), 1.0) - P_merw).max() < TOL


def test_degree_biased_is_log_degree(A):
    a = 1.7
    k = A.sum(1)
    P_db = A * k[None, :] ** a
    P_db /= P_db.sum(1, keepdims=True)
    assert np.abs(model.kernel(A, np.log(k), a) - P_db).max() < TOL


@pytest.mark.parametrize("lam", [0.0, 2.0])
def test_hitting_time_matches_monte_carlo(A, lam):
    g = model.rho_field(A, 0.0, seed=0)
    T = model.target_of(g)
    P = model.kernel(A, g, lam)
    mc, se = model.mfpt_monte_carlo(P, T, n_walks=2000)
    assert abs(model.mfpt_to_set(P, T) - mc) < 4 * se


def test_results_file_has_every_experiment():
    import json

    res = json.loads(model.RESULTS_PATH.read_text())
    assert set(res) == {"lams", "A", "B", "C", "D", "E", "F"}
    assert res["lams"] == [float(x) for x in model.LAMS]


def test_mouse_backbone_loads():
    A, vol, fa = model.load_mouse()
    assert A.shape == (332, 332) and len(vol) == len(fa) == 332
    B = model.knn_backbone(A, 6)
    assert np.array_equal(B, B.T) and (B > 0).sum(1).min() >= 6


def test_page_controls_render():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    at.switch_page("app_pages/network_science/gravity_mass.py").run()
    assert not at.exception, at.exception

    def widget(key):
        return next(w for w in at.sidebar if getattr(w, "key", None) == key)

    for key, value in [
        ("gm_lam", 6.0),
        ("gm_field", "heat"),
        ("gm_graph", "mouse"),
        ("gm_field", "volume"),
        ("gm_graph", "ER"),  # volume is mouse-only, so the field falls back to the ρ dial
    ]:
        widget(key).set_value(value)
        at.run()
        assert not at.exception, (key, value, at.exception)
    assert widget("gm_field").value == "rho"
    assert len(at.metric) == 4


def test_sparse_kernel_and_hitting_match_dense(A):
    import scipy.sparse as sp

    g = model.rho_field(A, 0.0, seed=0)
    T = model.target_of(g)
    for lam in (0.0, 3.0, 8.0):
        P = model.kernel(A, g, lam)
        Ps = model.kernel_sparse(sp.csr_matrix(A), g, lam)
        assert np.abs(Ps.toarray() - P).max() < TOL
        h, o = model.hitting_and_occupancy(Ps, T)
        dense = model.mfpt_to_set(P, T)
        assert abs(h.mean() - dense) / dense < 1e-9
        assert abs(o.sum() - h.mean()) / h.mean() < 1e-9  # occupancy adds up to hitting time


def test_well_basins_end_in_wells(A):
    g = model.rho_field(A, 0.0, seed=0)
    T = model.target_of(g)
    lab = model.well_basins(A, g, T)
    nbr_max = np.where(A > 0, g[None, :], -np.inf).max(axis=1)
    ends = np.unique(lab)
    assert np.all(T[ends] | (g[ends] > nbr_max[ends]))  # every basin is T or a local well
    assert np.all(lab[T] == np.flatnonzero(T))


def test_participation_ratio_limits():
    assert model.participation_ratio([1, 1, 1, 1]) == pytest.approx(0.25)
    assert model.participation_ratio([5, 0, 0]) == 1.0
