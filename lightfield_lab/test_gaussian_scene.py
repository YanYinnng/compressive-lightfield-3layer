from pathlib import Path
import unittest

import numpy as np

from .gaussian_scene import load_gaussian_ply, native_viewmats, stereo_poses


class CollectionTests(unittest.TestCase):
    def test_stereo_poses_have_metric_baseline_and_look_at_origin(self):
        for distance,yaw,elevation in [(.65,-35,-10),(1,0,0),(1.35,35,10)]:
            views=stereo_poses(distance,yaw,elevation)
            self.assertAlmostEqual(np.linalg.norm(views[1,:3,3]-views[0,:3,3]),.06,places=12)
            for view in views:
                np.testing.assert_allclose(view[:3,:3].T@view[:3,:3],np.eye(3),atol=1e-12)
                local=np.linalg.inv(view)@np.array([0,0,0,1])
                np.testing.assert_allclose(local[:2],[0,0],atol=1e-12)
                self.assertLess(local[2],0)

    def test_camera_conversion_projects_scene_center_to_principal_point(self):
        center=np.array([.2,-.3,.5]);scale=.12
        matrices=native_viewmats(stereo_poses(.9,20,8),scale,center)
        for matrix in matrices:
            point=matrix@np.r_[center,1]
            np.testing.assert_allclose(point[:2],[0,0],atol=1e-12)
            self.assertGreater(point[2],0)

    def test_render_accepts_a_single_camera(self):
        import torch
        from .gaussian_scene import GaussianTargetRenderer
        renderer=GaussianTargetRenderer.__new__(GaussianTargetRenderer)  # no gsplat: fake rasterizer
        renderer.device,renderer.scale_m,renderer.center='cpu',.12,np.zeros(3)
        renderer.means=renderer.quats=renderer.scales=renderer.opacities=renderer.sh=None
        renderer.metadata={'sh_degree':3}
        def fake(*args,**kwargs):
            count=len(args[5])
            return torch.full((count,4,6,3),.25),torch.full((count,4,6,1),.5),{}
        renderer.rasterization=fake
        for views in (stereo_poses(.9,0,0)[:1],stereo_poses(.9,0,0)):
            images,diagnostic=renderer.render(views,height=4,width=6)
            self.assertEqual(images.shape,(len(views),4,6,3))
            np.testing.assert_array_equal(images,191)  # .25+(1-.5) -> 0.75*255 rounded
        self.assertEqual(diagnostic['left_right_mean_abs_difference'],0.)
        self.assertIsNone(renderer.render(stereo_poses(.9,0,0)[:1],height=4,width=6)[1]['left_right_mean_abs_difference'])

    def test_supplied_ply_has_complete_degree3_gaussian_fields(self):
        p=Path(__file__).resolve().parents[1]/'data_collection/assets/lego_bulldozer.ply'
        source=load_gaussian_ply(p)
        self.assertEqual(len(source['means']),340889)
        self.assertEqual(source['sh'].shape,(340889,16,3))
        self.assertEqual(source['sh_degree'],3)
        self.assertTrue(np.isfinite(source['log_scales']).all())


if __name__=='__main__':unittest.main(verbosity=2)
