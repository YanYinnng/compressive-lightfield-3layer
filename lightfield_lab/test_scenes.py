import math
import unittest

import torch

from .scenes import analytic_case, camera_rays, render_scene


class AnalyticSceneTests(unittest.TestCase):
    def test_opaque_near_surface_occludes_the_middle_surface(self):
        # Optical axis along -world X; centre ray keeps Y=-4 and Z=0.
        view = torch.eye(4)
        view[:3, :3] = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        view[:3, 3] = torch.tensor([90., -4., 0.])
        target = render_scene(view[None], 1, 1, seed=0)[0, :, 0, 0]
        expected = torch.tensor([.75, .25, .18])+.14*math.sin(-4*1.3)
        torch.testing.assert_close(target, expected, atol=1e-6, rtol=1e-6)

    def test_center_ray_obeys_camera_rotation_and_translation(self):
        view = torch.eye(4)[None]
        view[0, :3, 3] = torch.tensor([1., 2., 3.])
        origins, directions = camera_rays(view, 1, 1)
        torch.testing.assert_close(origins[0, 0, 0], torch.tensor([1., 2., 3.]))
        torch.testing.assert_close(directions[0, 0, 0], torch.tensor([0., 0., -1.]))

    def test_neighbours_are_separate_small_displacements_with_known_targets(self):
        case = analytic_case(0, 36, 64, 'cpu')
        self.assertEqual(case['target'].shape, (2, 3, 36, 64))
        self.assertEqual(case['neighbour_target'].shape, (16, 3, 36, 64))
        for index, description in enumerate(case['metadata']['neighbours']):
            eye = description['eye']
            displacement = case['neighbour_views'][index, :3, 3]-case['views'][eye, :3, 3]
            self.assertAlmostEqual(float(displacement.norm()), description['radius_cm'], places=5)
        self.assertFalse(case['metadata']['neighbour_targets_used_for_optimization'])
        self.assertGreater(float(case['target'].std()), .1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
