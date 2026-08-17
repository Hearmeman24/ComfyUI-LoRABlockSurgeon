"""No network, no ComfyUI import, no GPU. Pure math and pure dict manipulation.

The tests that matter most are the two identity tests: they check the fast norm
against an explicitly materialised product. If those drift, every number this
tool prints is wrong and every pruning decision made from them is wrong.
"""

import os
import re
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import block_surgeon as B  # noqa: E402


class RelClose(unittest.TestCase):
    """Norms here run into the thousands, so the tolerance has to be RELATIVE.

    The fast path sums 1024 products of large fp32 numbers while the reference sums
    131072 squares; they agree to ~1e-6 relative, which is fp32 accumulation order,
    not a defect. Asserting an absolute `places=3` on a value of 2064 was the bug.
    """

    def assertRelClose(self, got, want, rel=1e-4, msg=None):
        denom = max(abs(want), 1e-12)
        self.assertLess(abs(got - want) / denom, rel,
                        msg or f"{got!r} vs {want!r} (rel tol {rel})")


class TestBlockIndex(unittest.TestCase):
    def test_matches_the_three_real_layouts(self):
        """Key shapes taken verbatim from real reference files."""
        self.assertEqual(B.block_index("diffusion_model.blocks.0.attn.wk.lora_A.weight"), 0)
        self.assertEqual(
            B.block_index("diffusion_model.transformer_blocks.47.attn1.to_k.lora_A.weight"), 47)
        self.assertEqual(B.block_index("diffusion_model.blocks.39.cross_attn.k.lora_B.weight"), 39)

    def test_matches_underscore_and_single_double_variants(self):
        self.assertEqual(B.block_index("lora_unet_single_blocks_12_linear1.lora_up.weight"), 12)
        self.assertEqual(B.block_index("double_blocks.7.img_attn.qkv.lora_A.weight"), 7)

    def test_unblocked_keys_are_none(self):
        for k in ("diffusion_model.final_layer.linear.lora_A.weight",
                  "diffusion_model.x_embedder.lora_B.weight",
                  "some.alpha"):
            self.assertIsNone(B.block_index(k), k)

    def test_multi_digit_index_is_not_truncated(self):
        self.assertEqual(B.block_index("diffusion_model.blocks.123.attn.lora_A.weight"), 123)


class TestParseBlockSpec(unittest.TestCase):
    def test_range(self):
        self.assertEqual(B.parse_block_spec("31-35"), {31, 32, 33, 34, 35})

    def test_list_and_mixed_with_spaces(self):
        self.assertEqual(B.parse_block_spec("0-2, 31 , 35"), {0, 1, 2, 31, 35})

    def test_single(self):
        self.assertEqual(B.parse_block_spec("7"), {7})

    def test_empty_is_empty_not_wildcard(self):
        """An empty spec must never mean 'everything' -- that would silently apply
        the whole LoRA when the operator meant to select nothing."""
        self.assertEqual(B.parse_block_spec(""), set())
        self.assertEqual(B.parse_block_spec("   "), set())

    def test_trailing_comma_is_tolerated(self):
        self.assertEqual(B.parse_block_spec("31,32,"), {31, 32})

    def test_garbage_raises(self):
        for bad in ("abc", "1-x", "3-1", "1..3"):
            with self.assertRaises(ValueError, msg=bad):
                B.parse_block_spec(bad)


def _lora_layer(sd, prefix, out_f, in_f, rank, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, in_f, generator=g)   # lora_A == "down"
    b = torch.randn(out_f, rank, generator=g)  # lora_B == "up"
    sd[f"{prefix}.lora_A.weight"] = a
    sd[f"{prefix}.lora_B.weight"] = b
    return b @ a


class TestLoRANorm(RelClose):
    def test_fast_norm_equals_the_materialised_product(self):
        """||B@A||_F via the 32x32 trace trick == building the full delta."""
        sd = {}
        delta = _lora_layer(sd, "diffusion_model.blocks.3.attn.wq", 512, 256, 32, seed=1)
        stats, skipped = B.layer_stats(sd)
        self.assertEqual(skipped, [])
        self.assertEqual(len(stats), 1)
        self.assertRelClose(stats[0].norm, float(torch.linalg.norm(delta)))

    def test_scale_ambiguity_does_not_change_the_measurement(self):
        """Multiply up by 10 and divide down by 10 -- identical delta, so the norm
        must be identical. This is the reason we measure the product and not the
        stored factors."""
        sd = {}
        _lora_layer(sd, "diffusion_model.blocks.3.attn.wq", 512, 256, 32, seed=2)
        first = B.layer_stats(sd)[0][0].norm
        sd["diffusion_model.blocks.3.attn.wq.lora_B.weight"] *= 10.0
        sd["diffusion_model.blocks.3.attn.wq.lora_A.weight"] /= 10.0
        self.assertRelClose(B.layer_stats(sd)[0][0].norm, first)

    def test_alpha_scales_by_alpha_over_rank(self):
        sd = {}
        delta = _lora_layer(sd, "diffusion_model.blocks.1.attn.wq", 128, 64, 16, seed=3)
        sd["diffusion_model.blocks.1.attn.wq.alpha"] = torch.tensor(8.0)
        expected = float(torch.linalg.norm(delta)) * (8.0 / 16)
        self.assertRelClose(B.layer_stats(sd)[0][0].norm, expected)

    def test_no_alpha_means_scale_one(self):
        """Verified: AI-Toolkit files carry no alpha tensor, so this is the
        path that actually runs on his LoRAs."""
        sd = {}
        delta = _lora_layer(sd, "diffusion_model.blocks.1.attn.wq", 128, 64, 16, seed=4)
        self.assertRelClose(B.layer_stats(sd)[0][0].norm, float(torch.linalg.norm(delta)))

    def test_kohya_up_down_naming_is_recognised(self):
        g = torch.Generator().manual_seed(5)
        a = torch.randn(8, 64, generator=g)
        b = torch.randn(32, 8, generator=g)
        sd = {"lora_unet_blocks_4_attn.lora_down.weight": a,
              "lora_unet_blocks_4_attn.lora_up.weight": b}
        stats, skipped = B.layer_stats(sd)
        self.assertEqual(skipped, [])
        self.assertRelClose(stats[0].norm, float(torch.linalg.norm(b @ a)))

    def test_lora_mid_is_reported_not_silently_zeroed(self):
        sd = {}
        _lora_layer(sd, "diffusion_model.blocks.2.conv", 64, 32, 8, seed=6)
        sd["diffusion_model.blocks.2.conv.lora_mid.weight"] = torch.zeros(8, 8, 3, 3)
        stats, skipped = B.layer_stats(sd)
        self.assertEqual(stats, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("lora_mid", skipped[0])


class TestLoKrNorm(RelClose):
    """Composition mirrors comfy/weight_adapter/lokr.py:59-86 -- delta = kron(w1, w2)."""

    def test_whole_factors_match_the_materialised_kron(self):
        g = torch.Generator().manual_seed(7)
        w1 = torch.randn(4, 4, generator=g)
        w2 = torch.randn(16, 12, generator=g)
        sd = {"diffusion_model.blocks.31.ff.lokr_w1": w1,
              "diffusion_model.blocks.31.ff.lokr_w2": w2}
        expected = float(torch.linalg.norm(torch.kron(w1, w2)))
        stats, skipped = B.layer_stats(sd)
        self.assertEqual(skipped, [])
        self.assertRelClose(stats[0].norm, expected)
        self.assertEqual(stats[0].kind, "lokr")

    def test_factored_w2_matches_the_materialised_kron(self):
        g = torch.Generator().manual_seed(8)
        w1 = torch.randn(4, 4, generator=g)
        w2a = torch.randn(16, 6, generator=g)
        w2b = torch.randn(6, 12, generator=g)
        sd = {"diffusion_model.blocks.31.ff.lokr_w1": w1,
              "diffusion_model.blocks.31.ff.lokr_w2_a": w2a,
              "diffusion_model.blocks.31.ff.lokr_w2_b": w2b}
        expected = float(torch.linalg.norm(torch.kron(w1, w2a @ w2b)))
        self.assertRelClose(B.layer_stats(sd)[0][0].norm, expected)

    def test_both_sides_factored(self):
        g = torch.Generator().manual_seed(9)
        w1a, w1b = torch.randn(4, 2, generator=g), torch.randn(2, 4, generator=g)
        w2a, w2b = torch.randn(16, 6, generator=g), torch.randn(6, 12, generator=g)
        sd = {"diffusion_model.blocks.31.ff.lokr_w1_a": w1a,
              "diffusion_model.blocks.31.ff.lokr_w1_b": w1b,
              "diffusion_model.blocks.31.ff.lokr_w2_a": w2a,
              "diffusion_model.blocks.31.ff.lokr_w2_b": w2b}
        expected = float(torch.linalg.norm(torch.kron(w1a @ w1b, w2a @ w2b)))
        self.assertRelClose(B.layer_stats(sd)[0][0].norm, expected)

    def test_half_a_factored_pair_is_reported_not_guessed(self):
        sd = {"diffusion_model.blocks.31.ff.lokr_w1": torch.randn(4, 4),
              "diffusion_model.blocks.31.ff.lokr_w2_a": torch.randn(16, 6)}
        stats, skipped = B.layer_stats(sd)
        self.assertEqual(stats, [])
        self.assertEqual(len(skipped), 1)


class TestUnsupportedIsNeverZero(unittest.TestCase):
    def test_loha_is_reported_as_unmeasured(self):
        """A silent zero would make the block look prunable when we simply did not
        understand it. That is the one failure mode that makes this tool harmful."""
        sd = {"diffusion_model.blocks.5.ff.hada_w1_a": torch.randn(8, 4),
              "diffusion_model.blocks.5.ff.hada_w1_b": torch.randn(4, 8)}
        stats, skipped = B.layer_stats(sd)
        self.assertEqual(stats, [])
        self.assertTrue(any("loha" in s for s in skipped))

    def test_skipped_layers_are_named_in_the_report(self):
        sd = {"diffusion_model.blocks.5.ff.hada_w1_a": torch.randn(8, 4),
              "diffusion_model.blocks.5.ff.hada_w1_b": torch.randn(4, 8)}
        report = B.format_report(sd)
        self.assertIn("skipped", report.lower() + " ")


class TestDiffFormat(RelClose):
    def test_direct_diff_tensor(self):
        d = torch.randn(32, 16)
        sd = {"diffusion_model.blocks.9.ff.diff": d}
        stats, skipped = B.layer_stats(sd)
        self.assertEqual(skipped, [])
        self.assertEqual(stats[0].kind, "diff")
        self.assertRelClose(stats[0].norm, float(torch.linalg.norm(d)))


class TestBlockStats(RelClose):
    def _two_block_sd(self):
        sd = {}
        d1 = _lora_layer(sd, "diffusion_model.blocks.0.attn.wq", 64, 32, 8, seed=10)
        d2 = _lora_layer(sd, "diffusion_model.blocks.0.attn.wk", 64, 32, 8, seed=11)
        d3 = _lora_layer(sd, "diffusion_model.blocks.5.attn.wq", 64, 32, 8, seed=12)
        d4 = _lora_layer(sd, "diffusion_model.final_layer.lin", 64, 32, 8, seed=13)
        return sd, d1, d2, d3, d4

    def test_block_aggregate_is_quadrature_not_a_plain_sum(self):
        """Two layers in one block aggregate as sqrt(n1^2+n2^2) -- the norm of the
        stacked delta. A plain sum would over-rank blocks that merely have more
        adapted layers."""
        sd, d1, d2, _, _ = self._two_block_sd()
        stats, _ = B.block_stats(sd)
        b0 = next(s for s in stats if s.block == 0)
        n1, n2 = (float(torch.linalg.norm(x)) for x in (d1, d2))
        self.assertRelClose(b0.norm, (n1 ** 2 + n2 ** 2) ** 0.5)
        self.assertNotAlmostEqual(b0.norm, n1 + n2, places=3)
        self.assertEqual(b0.layers, 2)

    def test_unblocked_tensors_get_their_own_group_labelled_not_dropped(self):
        sd, _, _, _, _ = self._two_block_sd()
        stats, _ = B.block_stats(sd)
        un = [s for s in stats if s.block is None]
        self.assertEqual(len(un), 1)
        self.assertEqual(un[0].label, "unblocked")

    def test_unblocked_group_sorts_last(self):
        sd, _, _, _, _ = self._two_block_sd()
        stats, _ = B.block_stats(sd)
        self.assertIsNone(stats[-1].block)
        self.assertEqual([s.block for s in stats[:-1]], [0, 5])


class TestFilterStateDict(unittest.TestCase):
    def _sd(self):
        sd = {}
        for b in (0, 1, 31, 32, 35):
            _lora_layer(sd, f"diffusion_model.blocks.{b}.attn.wq", 32, 16, 4, seed=b + 100)
        _lora_layer(sd, "diffusion_model.final_layer.lin", 32, 16, 4, seed=999)
        return sd

    def test_keep_mode_retains_only_the_selection(self):
        out, rep = B.filter_state_dict(self._sd(), {31, 32, 35}, "keep")
        self.assertEqual(rep["kept_blocks"], {31, 32, 35})
        self.assertEqual(rep["dropped_blocks"], {0, 1})
        self.assertEqual({B.block_index(k) for k in out} - {None}, {31, 32, 35})

    def test_drop_mode_is_the_complement(self):
        out, rep = B.filter_state_dict(self._sd(), {0, 1}, "drop")
        self.assertEqual(rep["dropped_blocks"], {0, 1})
        self.assertEqual({B.block_index(k) for k in out} - {None}, {31, 32, 35})

    def test_unblocked_tensors_survive_both_modes(self):
        """Embedders and heads are not blocks. Dropping them would change the
        adapter in a way the block spec never asked for."""
        for mode, sel in (("keep", {31}), ("drop", {0, 1, 31, 32, 35})):
            out, rep = B.filter_state_dict(self._sd(), sel, mode)
            self.assertIn("diffusion_model.final_layer.lin.lora_A.weight", out, mode)
            self.assertEqual(rep["unblocked_tensors"], 2, mode)

    def test_source_dict_is_never_mutated(self):
        sd = self._sd()
        before = set(sd)
        B.filter_state_dict(sd, {31}, "keep")
        self.assertEqual(set(sd), before)

    def test_empty_selection_in_keep_mode_leaves_only_unblocked(self):
        out, _ = B.filter_state_dict(self._sd(), set(), "keep")
        self.assertEqual({B.block_index(k) for k in out}, {None})

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            B.filter_state_dict(self._sd(), {1}, "prune")


class TestFormatReport(unittest.TestCase):
    def test_empty_dict_says_so_rather_than_printing_an_empty_table(self):
        self.assertIn("No measurable", B.format_report({}))

    def test_report_names_blocks_and_the_90pct_concentration(self):
        sd = {}
        # block 31 carries a much larger delta than the rest
        for b in (0, 1, 2):
            _lora_layer(sd, f"diffusion_model.blocks.{b}.attn.wq", 32, 16, 4, seed=b + 200)
        big = _lora_layer(sd, "diffusion_model.blocks.31.attn.wq", 32, 16, 4, seed=301)
        sd["diffusion_model.blocks.31.attn.wq.lora_B.weight"] *= 50.0
        del big
        report = B.format_report(sd)
        self.assertIn("31", report)
        self.assertIn("90% of the energy", report)
        self.assertIn("energy%", report)

    def test_sort_by_norm_puts_the_largest_first(self):
        sd = {}
        _lora_layer(sd, "diffusion_model.blocks.0.attn.wq", 32, 16, 4, seed=400)
        _lora_layer(sd, "diffusion_model.blocks.9.attn.wq", 32, 16, 4, seed=401)
        sd["diffusion_model.blocks.9.attn.wq.lora_B.weight"] *= 40.0
        # Table rows only: the block column is right-aligned in width 9, so a row
        # always starts with whitespace. The summary header ("2 groups | ...") does
        # not, and picking it up was what made the first version of this test lie.
        rows = [l for l in B.format_report(sd, sort_by="norm").splitlines()
                if re.match(r"^\s+\d+\s+\d+\s+[\d.]+", l)]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].split()[0], "9")


class TestCompact(unittest.TestCase):
    def test_runs_are_collapsed(self):
        self.assertEqual(B._compact([31, 32, 33, 35]), "31-33,35")
        self.assertEqual(B._compact([5]), "5")
        self.assertEqual(B._compact([]), "(none)")
        self.assertEqual(B._compact([1, 3, 5]), "1,3,5")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestBlocksPresent(unittest.TestCase):
    """Backs the filter node's typo warning. If this under-reports, a spec naming
    a block the file does not have passes silently and the operator believes a
    prune happened that did not."""

    def test_lists_every_numbered_block_once(self):
        sd = {}
        for b in (0, 3, 31):
            _lora_layer(sd, f"diffusion_model.blocks.{b}.attn.wq", 16, 8, 2, seed=b + 500)
        _lora_layer(sd, "diffusion_model.final_layer.lin", 16, 8, 2, seed=600)
        self.assertEqual(B.blocks_present(sd), {0, 3, 31})

    def test_empty_dict_has_no_blocks(self):
        self.assertEqual(B.blocks_present({}), set())

    def test_a_spec_outside_the_file_is_detectable(self):
        sd = {}
        _lora_layer(sd, "diffusion_model.blocks.5.attn.wq", 16, 8, 2, seed=700)
        self.assertEqual(B.parse_block_spec("31-35") - B.blocks_present(sd),
                         {31, 32, 33, 34, 35})
