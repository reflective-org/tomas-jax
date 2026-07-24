"""Vectorized MNFIX vs the sequential Fortran-faithful reference."""
import numpy as np

import jax.numpy as jnp

from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.fast.config import NBINS, xk_boundaries
from tomas_jax.fast.mnfix import mnfix

XK = xk_boundaries()
ICOMP_NODIAG = 1


def _rand_state(rng, drift=None, max_up=6.0, min_down=0.2):
    """Random 2-component state; drift='up'/'down' pushes avg out of range."""
    Nk = jnp.asarray(rng.uniform(1.0, 1e6, NBINS))
    xmid = np.sqrt(np.asarray(XK[:-1] * XK[1:]))
    factor = np.ones(NBINS)
    if drift == "up":
        factor = rng.uniform(1.0, max_up, NBINS)
    elif drift == "down":
        factor = rng.uniform(min_down, 1.0, NBINS)
    mso4 = np.asarray(Nk) * xmid * factor
    mh2o = mso4 * rng.uniform(0.0, 2.0, NBINS)
    Mk = jnp.stack([jnp.asarray(mso4), jnp.asarray(mh2o)], axis=1)
    return Nk, Mk


def _seq(Nk, Mk):
    return mnfix_jax(Nk, Mk, XK, icomp_nodiag=ICOMP_NODIAG)


def _fast(Nk, Mk):
    return mnfix(Nk, Mk, XK, icomp_nodiag=ICOMP_NODIAG)


class TestPhases12Exact:
    def test_empty_bins_bitexact(self):
        rng = np.random.default_rng(1)
        Nk, Mk = _rand_state(rng)
        Nk = Nk.at[::4].set(0.0)  # empty every 4th bin
        Ns, Ms = _seq(Nk, Mk)
        Nf, Mf = _fast(Nk, Mk)
        np.testing.assert_array_equal(np.asarray(Nf), np.asarray(Ns))
        np.testing.assert_array_equal(np.asarray(Mf), np.asarray(Ms))

    def test_in_range_state_is_noop(self):
        rng = np.random.default_rng(2)
        Nk, Mk = _rand_state(rng)  # avg == geometric mean: in range
        Nf, Mf = _fast(Nk, Mk)
        np.testing.assert_allclose(np.asarray(Nf), np.asarray(Nk), rtol=1e-15)
        np.testing.assert_allclose(np.asarray(Mf), np.asarray(Mk), rtol=1e-15)


class TestPhase3:
    def test_isolated_drift_matches_sequential(self):
        """A single drifted bin (no receiver collisions) is bit-equal."""
        rng = np.random.default_rng(3)
        for drifted_bin, factor in [(5, 3.0), (20, 4.5), (10, 0.4)]:
            Nk, Mk = _rand_state(rng)
            Mk = Mk.at[drifted_bin, :].mul(factor)
            Ns, Ms = _seq(Nk, Mk)
            Nf, Mf = _fast(Nk, Mk)
            np.testing.assert_allclose(np.asarray(Nf), np.asarray(Ns), rtol=1e-14)
            np.testing.assert_allclose(np.asarray(Mf), np.asarray(Ms), rtol=1e-14)

    def test_conserves_number_and_mass(self):
        """Phase 3 conserves totals exactly (states that pass phases 1-2).

        Drift is bounded to the strictly conservative regime: up-factor <= 2
        keeps every avg below the global grid ceiling xk[nbins] (larger drift
        near the top bins trips phase 2, which trims mass/number by design —
        Fortran behavior), and the edge bins are pinned in-range.
        """
        rng = np.random.default_rng(4)
        xmid = jnp.sqrt(XK[:-1] * XK[1:])
        for drift in ["up", "down", None]:
            Nk, Mk = _rand_state(rng, drift=drift, max_up=2.0, min_down=0.75)
            Mk = Mk.at[0, 0].set(float(Nk[0] * xmid[0]))
            Mk = Mk.at[-1, 0].set(float(Nk[-1] * xmid[-1]))
            Nf, Mf = _fast(Nk, Mk)
            np.testing.assert_allclose(
                float(jnp.sum(Nf)), float(jnp.sum(Nk)), rtol=1e-13
            )
            np.testing.assert_allclose(
                float(jnp.sum(Mf[:, 0])), float(jnp.sum(Mk[:, 0])), rtol=1e-13
            )

    def test_sequential_also_agrees_on_totals(self):
        rng = np.random.default_rng(5)
        Nk, Mk = _rand_state(rng, drift="up")
        Ns, Ms = _seq(Nk, Mk)
        Nf, Mf = _fast(Nk, Mk)
        np.testing.assert_allclose(
            float(jnp.sum(Nf)), float(jnp.sum(Ns)), rtol=1e-13
        )
        np.testing.assert_allclose(
            float(jnp.sum(Mf[:, 0])), float(jnp.sum(Ms[:, 0])), rtol=1e-13
        )

    def test_idempotent_after_two_sweeps(self):
        """A second full mnfix application is a no-op: all fixable drift fixed."""
        rng = np.random.default_rng(6)
        for drift in ["up", "down"]:
            Nk, Mk = _rand_state(rng, drift=drift)
            N1, M1 = _fast(Nk, Mk)
            N2, M2 = _fast(N1, M1)
            np.testing.assert_allclose(np.asarray(N2), np.asarray(N1), rtol=1e-12)
            np.testing.assert_allclose(np.asarray(M2), np.asarray(M1), rtol=1e-12)

    def test_adversarial_multibin_jump(self):
        """Nucleation-like surge: bin 0 mass belongs 12 bins up."""
        Nk = jnp.full(NBINS, 1.0)
        xmid = jnp.sqrt(XK[:-1] * XK[1:])
        Mk = jnp.stack([Nk * xmid, jnp.zeros(NBINS)], axis=1)
        Mk = Mk.at[0, 0].set(float(XK[12] * 1.05))  # huge avg in bin 0
        Nf, Mf = _fast(Nk, Mk)
        Ns, Ms = _seq(Nk, Mk)
        # Deposit must land in the same target bin with the same magnitude
        np.testing.assert_allclose(np.asarray(Nf), np.asarray(Ns), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(Mf), np.asarray(Ms), rtol=1e-12)


class TestBatched:
    def test_batched_equals_percell(self):
        rng = np.random.default_rng(7)
        states = [_rand_state(rng, drift=d) for d in [None, "up", "down", "up"]]
        NkB = jnp.stack([s[0] for s in states])
        MkB = jnp.stack([s[1] for s in states])
        NfB, MfB = mnfix(NkB, MkB, XK, icomp_nodiag=ICOMP_NODIAG)
        for c, (Nk, Mk) in enumerate(states):
            Nf, Mf = _fast(Nk, Mk)
            np.testing.assert_array_equal(np.asarray(NfB[c]), np.asarray(Nf))
            np.testing.assert_array_equal(np.asarray(MfB[c]), np.asarray(Mf))
