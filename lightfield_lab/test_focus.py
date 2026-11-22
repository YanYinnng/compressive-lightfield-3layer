import unittest

import torch

from .core import CachedRenderer, synthetic_geometry
from .focus import FocusGeometry, focal_stack, focal_stack_many, focus_metrics, lcd_rays, pupil_offsets
from .scenes import camera_rays


class FocusTests(unittest.TestCase):
    def test_zero_pupil_matches_cached_pinhole_renderer_and_mask(self):
        torch.manual_seed(3)
        views, screens = synthetic_geometry(36, 64)
        patterns = torch.rand(3, 3, 36, 64)*.4
        original = CachedRenderer(views, screens, 36, 64)
        origins, directions = camera_rays(views, 36, 64)
        actual, mask = lcd_rays(patterns, screens, origins, directions)
        torch.testing.assert_close(actual, original(patterns), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(mask.float(), original.geometry_mask)

    def test_focus_plane_is_invariant_across_pupil_samples(self):
        views, _ = synthetic_geometry(9, 16)
        geometry = FocusGeometry(views[0], 9, 16, diameter_mm=6, rings=3)
        origins, directions = geometry.rays(80)
        hits = origins+80*directions
        torch.testing.assert_close(hits, hits[:1].expand_as(hits), atol=1e-5, rtol=1e-6)

    def test_known_textured_plane_focuses_at_its_geometric_distance(self):
        view = torch.eye(4)
        geometry = FocusGeometry(view, 36, 64, diameter_mm=6, rings=4)
        images = []
        for distance in [60., 80., 100.]:
            origins, directions = geometry.rays(distance)
            t = (-80-origins[..., 2])/directions[..., 2]
            point = origins+t[..., None]*directions
            images.append((.5+.4*torch.sin(point[..., 0]*45)*torch.cos(point[..., 1]*37)).mean(0))
        _, center = camera_rays(view[None], 36, 64)
        point = center[0]*80
        sharp = .5+.4*torch.sin(point[..., 0]*45)*torch.cos(point[..., 1]*37)
        mse = torch.stack([(image-sharp).square().mean() for image in images])
        self.assertEqual(int(mse.argmin()), 1)
        self.assertLess(float(mse[1]), 1e-9)
        self.assertGreater(float(mse[[0, 2]].min()), .001)

    def test_aperture_quadrature_is_centered_and_has_correct_area_moment(self):
        points = pupil_offsets(6, 5)
        self.assertEqual(len(points), 150)
        torch.testing.assert_close(points.mean(0), torch.zeros(2), atol=1e-7, rtol=0)
        self.assertLess(float(points.norm(dim=1).max()), .3)
        self.assertAlmostEqual(float(points.square().sum(1).mean()), .3**2/2, places=5)

    def test_batched_methods_match_independent_frozen_pattern_integration(self):
        views, screens = synthetic_geometry(18, 32)
        geometry = FocusGeometry(views[0], 18, 32, diameter_mm=4, rings=2)
        patterns = torch.rand(2, 3, 3, 18, 32)*.4
        actual, masks = focal_stack_many(geometry, [80., 90.], screens, patterns)
        for index in range(2):
            expected, valid = focal_stack(geometry, [80., 90.], screens, patterns[index])
            torch.testing.assert_close(actual[index], expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(masks, valid)

    def test_focus_response_distinguishes_correct_and_focus_invariant_outputs(self):
        truth = torch.tensor([.3, .5, .7])[:, None, None, None].expand(3, 3, 8, 8)
        mask = torch.ones(1, 8, 8)
        arguments = dict(truth=truth, sharp=truth[1], valid=mask, rois={'near': mask}, distances_cm=[70, 80, 90])
        correct = focus_metrics(truth, **arguments)['rois']['near']
        flat = focus_metrics(truth.mean(0).expand_as(truth), **arguments)['rois']['near']
        self.assertAlmostEqual(correct['focus_response_nrmse'], 0.)
        self.assertAlmostEqual(correct['focus_response_gain'], 1.)
        self.assertAlmostEqual(flat['focus_response_nrmse'], 1.)
        self.assertAlmostEqual(flat['focus_response_gain'], 0.)

    def test_pinhole_has_no_focus_signal_and_no_identifiable_peak(self):
        views, _ = synthetic_geometry(18, 32)
        geometry = FocusGeometry(views[0], 18, 32, diameter_mm=0, rings=3)
        truth, _ = focal_stack(geometry, [80., 90., 100.], seed=0)
        torch.testing.assert_close(truth, truth[:1].expand_as(truth))
        mask = torch.ones(1, 18, 32)
        result = focus_metrics(truth, truth, truth[0], mask, {'near': mask}, [80., 90., 100.])
        self.assertFalse(result['rois']['near']['peak_observable'])
        self.assertIsNone(result['rois']['near']['peak_error_diopters'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
