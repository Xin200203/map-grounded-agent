import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, Dict, List, Optional, Sequence

from src.utils.fmm.depth_utils import (
    get_point_cloud_from_z_t,
    transform_camera_view_t,
    transform_pose_t,
    splat_feat_nd
)
from src.utils.model import get_grid, ChannelPool
from src.utils.camera import get_camera_matrix


SEMANTIC_CATEGORY_NAMES = (
    "chair",
    "couch",
    "potted plant",
    "bed",
    "toilet",
    "tv",
    "dining-table",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "cup",
    "bottle",
    "other",
)

MIN_FOOTPRINT_CELLS_BY_CATEGORY = {
    "chair": (8, 8),
    "couch": (10, 18),
    "potted plant": (6, 6),
    "bed": (16, 24),
    "toilet": (8, 8),
    "tv": (4, 10),
    "dining-table": (14, 18),
    "oven": (8, 8),
    "sink": (8, 8),
    "refrigerator": (12, 8),
    "book": (4, 4),
    "clock": (4, 4),
    "vase": (5, 5),
    "cup": (3, 3),
    "bottle": (3, 3),
    "cabinet": (10, 14),
    "windows": (4, 18),
    "mirror": (6, 10),
    "other": (6, 6),
}


class Mapping(nn.Module):
    def __init__(self, args):
        super(Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-80 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)
        self.vfov = np.arctan((self.screen_h/2.) / self.camera_matrix.f)
        self.min_vision = self.agent_height / np.tan(self.vfov) # cm

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last, agent_heights):
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = transform_camera_view_t(
            point_cloud_t, agent_heights * 100., 0, self.device)

        agent_view_centered_t = transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(35 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)
        floor_z = int(-35 / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)


        min_vision_std = int(self.min_vision // z_resolution)
        around_floor_proj = voxels[..., :min_z].sum(4) 
        under_floor_proj = voxels[..., floor_z:min_z].sum(4) 
        under_floor_proj = (under_floor_proj == 0.0).float()# have floor = 0, no floor or not deteced = 1
        under_floor_proj = under_floor_proj * around_floor_proj # no floor and detected = 1

        replace_element = torch.ones_like(depth[:, -1, :]) * self.min_vision
        re_depth = torch.where(depth[:, -1, :] < 3000, depth[:, -1, :], replace_element)
        count = ((re_depth - self.min_vision - 60) > 0).sum(dim=1)
        index = torch.nonzero(count > (re_depth.shape[1] / 4))

        under_floor_proj[index, 0:1, min_vision_std:min_vision_std+1, \
                (self.vision_range-6)//2 : (self.vision_range+6)//2] \
                    = 1.

        fp_map_pred = agent_height_proj[:, 0:1, :, :] + under_floor_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses



class BEV_Map():
    def __init__(self, args):
        self.args = args

        self.num_scenes = self.args.num_processes
        nc = self.args.num_sem_categories + 4  # num channels
        self.device = self.args.device

        self.map_size = self.args.map_size
        self.global_width = self.args.global_width
        self.global_height = self.args.global_height
        self.local_width = self.args.local_width
        self.local_height = self.args.local_height

        self.mapping_module = Mapping(self.args).to(self.device)
        self.mapping_module.eval()

        # Initializing full and local map
        self.full_map = torch.zeros(self.num_scenes, nc, self.global_width, self.global_height).float().to(self.device)
        self.local_map = torch.zeros(self.num_scenes, nc, self.local_width,
                                self.local_height).float().to(self.device)

        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.local_map_boundary = np.zeros((self.num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))
        self.last_semantic_projection_summary = {
            "schema_version": "smoothnav.semantic_projection_summary.v1",
            "envs": [],
        }
        self.last_semantic_instance_footprints = {
            "schema_version": "smoothnav.semantic_instance_footprints.v1",
            "envs": [],
        }

    def _semantic_category_label(self, category_index: int) -> str:
        try:
            idx = int(category_index)
        except Exception:
            return "other"
        if 0 <= idx < len(SEMANTIC_CATEGORY_NAMES):
            return str(SEMANTIC_CATEGORY_NAMES[idx])
        return "other"

    def _instances_from_infos(self, infos: MappingABC, env_idx: int) -> List[Dict[str, Any]]:
        if not isinstance(infos, MappingABC):
            return []
        payload = (
            infos.get("semantic_instances")
            or infos.get("smoothnav_semantic_instances")
            or infos.get("semantic_instance_footprints_input")
        )
        if payload is None:
            return []
        if isinstance(payload, MappingABC):
            envs = payload.get("envs")
            if isinstance(envs, SequenceABC):
                for env_payload in envs:
                    if isinstance(env_payload, MappingABC) and int(env_payload.get("env_idx", -1) or -1) == int(env_idx):
                        return [dict(item) for item in (env_payload.get("instances") or []) if isinstance(item, MappingABC)]
            return [dict(item) for item in (payload.get("instances") or []) if isinstance(item, MappingABC)]
        if isinstance(payload, SequenceABC) and not isinstance(payload, (str, bytes)):
            return [dict(item) for item in payload if isinstance(item, MappingABC)]
        return []

    def _clip_bbox_rc(self, bbox: Sequence[int], *, height: int, width: int) -> Optional[List[int]]:
        try:
            r0, c0, r1, c1 = [int(v) for v in bbox[:4]]
        except Exception:
            return None
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0
        r0 = max(0, min(int(height) - 1, r0))
        r1 = max(0, min(int(height) - 1, r1))
        c0 = max(0, min(int(width) - 1, c0))
        c1 = max(0, min(int(width) - 1, c1))
        if r1 < r0 or c1 < c0:
            return None
        return [r0, c0, r1, c1]

    def _expand_bbox_for_category(
        self,
        bbox: Sequence[int],
        *,
        category_label: str,
        height: int,
        width: int,
    ) -> Optional[List[int]]:
        clipped = self._clip_bbox_rc(bbox, height=height, width=width)
        if clipped is None:
            return None
        r0, c0, r1, c1 = clipped
        min_h, min_w = MIN_FOOTPRINT_CELLS_BY_CATEGORY.get(
            str(category_label or "other").strip().lower(),
            MIN_FOOTPRINT_CELLS_BY_CATEGORY["other"],
        )
        cur_h = int(r1 - r0 + 1)
        cur_w = int(c1 - c0 + 1)
        target_h = max(cur_h, int(min_h))
        target_w = max(cur_w, int(min_w))
        center_r = int(round((r0 + r1) / 2.0))
        center_c = int(round((c0 + c1) / 2.0))
        nr0 = center_r - target_h // 2
        nr1 = nr0 + target_h - 1
        nc0 = center_c - target_w // 2
        nc1 = nc0 + target_w - 1
        if nr0 < 0:
            nr1 -= nr0
            nr0 = 0
        if nc0 < 0:
            nc1 -= nc0
            nc0 = 0
        if nr1 >= height:
            delta = nr1 - height + 1
            nr0 -= delta
            nr1 -= delta
        if nc1 >= width:
            delta = nc1 - width + 1
            nc0 -= delta
            nc1 -= delta
        return self._clip_bbox_rc([nr0, nc0, nr1, nc1], height=height, width=width)

    def _shift_bbox_to_local_interior(
        self,
        bbox: Sequence[int],
        *,
        env_idx: int,
        max_shift: Optional[int] = None,
    ) -> Optional[List[int]]:
        base = self._clip_bbox_rc(
            bbox,
            height=int(self.local_map.shape[-2]),
            width=int(self.local_map.shape[-1]),
        )
        if base is None:
            return None
        try:
            explored = self.local_map[env_idx, 1].detach().cpu().numpy() > 0
            obstacle = self.local_map[env_idx, 0].detach().cpu().numpy() > 0
        except Exception:
            return base
        r0, c0, r1, c1 = base
        bh = int(r1 - r0 + 1)
        bw = int(c1 - c0 + 1)
        max_shift = int(max_shift if max_shift is not None else max(4, min(24, round(max(bh, bw) * 1.5))))
        best = base
        best_score = -1e18
        h, w = explored.shape
        for dr in range(-max_shift, max_shift + 1):
            nr0 = r0 + dr
            nr1 = nr0 + bh - 1
            if nr0 < 0 or nr1 >= h:
                continue
            for dc in range(-max_shift, max_shift + 1):
                nc0 = c0 + dc
                nc1 = nc0 + bw - 1
                if nc0 < 0 or nc1 >= w:
                    continue
                exp_patch = explored[nr0 : nr1 + 1, nc0 : nc1 + 1]
                obs_patch = obstacle[nr0 : nr1 + 1, nc0 : nc1 + 1]
                interior = np.logical_and(exp_patch, ~obs_patch)
                unknown = ~exp_patch
                score = (
                    3.0 * float(np.count_nonzero(interior))
                    - 5.0 * float(np.count_nonzero(obs_patch))
                    - 1.5 * float(np.count_nonzero(unknown))
                    - 0.04 * float(abs(dr) + abs(dc))
                )
                if score > best_score:
                    best_score = score
                    best = [int(nr0), int(nc0), int(nr1), int(nc1)]
        return best

    def _bbox_cells_full(self, bbox_local: Sequence[int], *, env_idx: int, max_cells: int = 12000) -> List[List[int]]:
        local = self._clip_bbox_rc(
            bbox_local,
            height=int(self.local_map.shape[-2]),
            width=int(self.local_map.shape[-1]),
        )
        if local is None:
            return []
        r0, c0, r1, c1 = local
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area <= 0 or area > int(max_cells):
            return []
        try:
            row_off = int(self.local_map_boundary[env_idx, 0])
            col_off = int(self.local_map_boundary[env_idx, 2])
        except Exception:
            row_off = 0
            col_off = 0
        h_full, w_full = int(self.full_map.shape[-2]), int(self.full_map.shape[-1])
        cells = []
        for r in range(r0, r1 + 1):
            fr = int(row_off + r)
            if fr < 0 or fr >= h_full:
                continue
            for c in range(c0, c1 + 1):
                fc = int(col_off + c)
                if 0 <= fc < w_full:
                    cells.append([fr, fc])
        return cells

    def _project_semantic_instance_footprints(
        self,
        obs: torch.Tensor,
        poses: torch.Tensor,
        poses_last: torch.Tensor,
        agent_heights: torch.Tensor,
        infos: MappingABC,
    ) -> Dict[str, Any]:
        """Project detector instance boxes into dense BEV footprint boxes.

        This keeps object footprints separate from ``full_map[4:]`` surface
        semantics.  The projected cells are stored as evidence for planner BEV
        rendering and replay; they do not mutate the navigation map channels.
        """

        summary = {
            "schema_version": "smoothnav.semantic_instance_footprints.v1",
            "envs": [],
        }
        if obs is None or obs.ndim != 4:
            self.last_semantic_instance_footprints = summary
            return summary
        mapping = self.mapping_module
        bs, channels, h, w = obs.size()
        for env_idx in range(self.num_scenes):
            instances = self._instances_from_infos(infos, env_idx)
            env_summary = {
                "env_idx": int(env_idx),
                "input_instance_count": int(len(instances)),
                "projected_instance_count": 0,
                "instances": [],
                "warnings": [],
            }
            if not instances or env_idx >= bs:
                summary["envs"].append(env_summary)
                continue
            masks = []
            records = []
            obs_e = obs[env_idx]
            for inst_idx, item in enumerate(instances, start=1):
                try:
                    cat_idx = int(item.get("category_index", item.get("class_id", -1)))
                except Exception:
                    cat_idx = -1
                if cat_idx < 0 or 4 + cat_idx >= channels:
                    continue
                bbox = item.get("bbox_xyxy") or item.get("bbox")
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in list(bbox)[:4]]
                except Exception:
                    continue
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                if x2 <= x1 or y2 <= y1:
                    continue
                mask = torch.zeros((h, w), dtype=torch.float32, device=self.device)
                semantic_channel = obs_e[4 + cat_idx] > 0
                semantic_crop = semantic_channel[y1 : y2 + 1, x1 : x2 + 1]
                if bool(torch.count_nonzero(semantic_crop).item() > 0):
                    mask[y1 : y2 + 1, x1 : x2 + 1] = semantic_crop.float()
                    seed_source = "semantic_mask_in_detector_bbox"
                else:
                    # Keep this as an explicit, low-confidence fallback: the box
                    # is detector evidence, but not a true projected semantic mask.
                    mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0
                    seed_source = "detector_bbox_depth_fallback"
                masks.append(mask)
                label = str(item.get("category_label") or self._semantic_category_label(cat_idx))
                records.append(
                    {
                        "id": str(item.get("id") or f"det_{inst_idx:03d}"),
                        "category_index": int(cat_idx),
                        "category_label": label,
                        "confidence": float(item.get("confidence", 0.0) or 0.0),
                        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "seed_source": seed_source,
                        "source_semantic_seed_pixel_count": int(item.get("semantic_seed_pixel_count", 0) or 0),
                    }
                )
            if not masks:
                summary["envs"].append(env_summary)
                continue
            try:
                depth = obs[env_idx : env_idx + 1, 3, :, :]
                point_cloud_t = get_point_cloud_from_z_t(
                    depth,
                    mapping.camera_matrix,
                    self.device,
                    scale=mapping.du_scale,
                )
                agent_view_t = transform_camera_view_t(
                    point_cloud_t,
                    agent_heights[env_idx : env_idx + 1] * 100.0,
                    0,
                    self.device,
                )
                agent_view_centered_t = transform_pose_t(
                    agent_view_t,
                    mapping.shift_loc,
                    self.device,
                )
                max_h = mapping.max_height
                min_h = mapping.min_height
                xy_resolution = mapping.resolution
                z_resolution = mapping.z_resolution
                vision_range = mapping.vision_range
                xyz = agent_view_centered_t.float()
                xyz[..., :2] = (xyz[..., :2] / xy_resolution)
                xyz[..., :2] = (xyz[..., :2] - vision_range // 2.0) / vision_range * 2.0
                xyz[..., 2] = xyz[..., 2] / z_resolution
                xyz[..., 2] = (xyz[..., 2] - (max_h + min_h) // 2.0) / (max_h - min_h) * 2.0
                pooled = nn.AvgPool2d(mapping.du_scale)(torch.stack(masks, dim=0).unsqueeze(0))
                feat = pooled.view(1, len(masks), h // mapping.du_scale * w // mapping.du_scale)
                xyz = xyz.permute(0, 3, 1, 2).contiguous().view(1, 3, -1)
                init_grid = torch.zeros(
                    1,
                    len(masks),
                    vision_range,
                    vision_range,
                    max_h - min_h,
                    device=self.device,
                )
                voxels = splat_feat_nd(init_grid, feat, xyz).transpose(2, 3)
                # Footprints should summarize where the object occupies the
                # top-down plane, not only the agent-height obstacle slice used
                # for collision.  Use all projected object mask support here;
                # otherwise tall/low surfaces can disappear from the footprint
                # path even when the detector saw a valid instance.
                instance_proj = voxels.sum(4)
                agent_view = torch.zeros(
                    1,
                    len(masks),
                    mapping.map_size_cm // mapping.resolution,
                    mapping.map_size_cm // mapping.resolution,
                    device=self.device,
                )
                x1 = mapping.map_size_cm // (mapping.resolution * 2) - vision_range // 2
                x2 = x1 + vision_range
                y1 = mapping.map_size_cm // (mapping.resolution * 2)
                y2 = y1 + vision_range
                agent_view[:, :, y1:y2, x1:x2] = torch.clamp(instance_proj, min=0.0, max=1.0)

                current_poses = poses_last[env_idx : env_idx + 1].clone()
                rel_pose = poses[env_idx : env_idx + 1]
                current_poses[:, 1] += rel_pose[:, 0] * torch.sin(current_poses[:, 2] / 57.29577951308232) + rel_pose[:, 1] * torch.cos(current_poses[:, 2] / 57.29577951308232)
                current_poses[:, 0] += rel_pose[:, 0] * torch.cos(current_poses[:, 2] / 57.29577951308232) - rel_pose[:, 1] * torch.sin(current_poses[:, 2] / 57.29577951308232)
                current_poses[:, 2] += rel_pose[:, 2] * 57.29577951308232
                current_poses[:, 2] = torch.fmod(current_poses[:, 2] - 180.0, 360.0) + 180.0
                current_poses[:, 2] = torch.fmod(current_poses[:, 2] + 180.0, 360.0) - 180.0
                st_pose = current_poses.clone().detach()
                st_pose[:, :2] = -(
                    st_pose[:, :2] * 100.0 / mapping.resolution
                    - mapping.map_size_cm // (mapping.resolution * 2)
                ) / (mapping.map_size_cm // (mapping.resolution * 2))
                st_pose[:, 2] = 90.0 - st_pose[:, 2]
                rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
                rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
                translated = F.grid_sample(rotated, trans_mat, align_corners=True)[0]
            except Exception as exc:
                env_summary["warnings"].append(f"projection_failed:{exc}")
                summary["envs"].append(env_summary)
                continue

            for rec_idx, record in enumerate(records):
                mask_np = translated[rec_idx].detach().cpu().numpy() > 0.01
                if not np.any(mask_np):
                    continue
                rr, cc = np.where(mask_np)
                raw_local_bbox = [int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())]
                # Mature semantic BEV maps are produced by projecting semantic
                # pixels/instances into top-down cells, then rasterizing those
                # cells.  Do not shift the projected support to a "nice" nearby
                # interior patch: that made object boxes detach from the actual
                # evidence and visibly slide along/away from walls.  A tight
                # bbox around the projected instance support is a deterministic
                # first-order footprint; expansion/repair, if needed, must be a
                # separately diagnosed upstream mapping problem.
                local_bbox = self._clip_bbox_rc(
                    raw_local_bbox,
                    height=int(self.local_map.shape[-2]),
                    width=int(self.local_map.shape[-1]),
                )
                if local_bbox is None:
                    continue
                cells = self._bbox_cells_full(local_bbox, env_idx=env_idx)
                if not cells:
                    continue
                try:
                    row_off = int(self.local_map_boundary[env_idx, 0])
                    col_off = int(self.local_map_boundary[env_idx, 2])
                except Exception:
                    row_off = 0
                    col_off = 0
                bbox_full = [
                    int(row_off + local_bbox[0]),
                    int(col_off + local_bbox[1]),
                    int(row_off + local_bbox[2]),
                    int(col_off + local_bbox[3]),
                ]
                raw_bbox_full = [
                    int(row_off + raw_local_bbox[0]),
                    int(col_off + raw_local_bbox[1]),
                    int(row_off + raw_local_bbox[2]),
                    int(col_off + raw_local_bbox[3]),
                ]
                env_summary["instances"].append(
                    {
                        "id": f"SI{len(env_summary['instances']) + 1:03d}",
                        "detector_id": record["id"],
                        "caption": record["category_label"],
                        "category_label": record["category_label"],
                        "category_index": int(record["category_index"]),
                        "confidence": float(record["confidence"]),
                        "bbox_rc": bbox_full,
                        "raw_surface_bbox_rc": raw_bbox_full,
                        "local_bbox_rc": list(local_bbox),
                        "raw_local_surface_bbox_rc": list(raw_local_bbox),
                        "footprint_source": "semantic_instance_depth_bbox_cells",
                        "footprint_confidence": 0.65 if record["seed_source"] == "semantic_mask_in_detector_bbox" else 0.45,
                        "footprint_pixel_count": int(len(cells)),
                        "footprint_seed_cell_count": int(len(rr)),
                        "footprint_rc_indices": cells,
                        "footprint_encoding": "rc_indices_from_projected_instance_bbox",
                        "bbox_expansion_applied": False,
                        "bbox_shift_applied": False,
                        "seed_source": record["seed_source"],
                        "source_semantic_seed_pixel_count": int(record["source_semantic_seed_pixel_count"]),
                        "source": "bev_map.semantic_instance_footprints",
                    }
                )
            env_summary["projected_instance_count"] = int(len(env_summary["instances"]))
            summary["envs"].append(env_summary)
        self.last_semantic_instance_footprints = summary
        return summary

    def semantic_instance_footprints(self, env_idx=0):
        summary = getattr(self, "last_semantic_instance_footprints", None) or {
            "schema_version": "smoothnav.semantic_instance_footprints.v1",
            "envs": [],
        }
        envs = list(summary.get("envs", []) or [])
        idx = int(env_idx or 0)
        if 0 <= idx < len(envs):
            return list(envs[idx].get("instances") or [])
        return []

    def _semantic_channel_counts(self, value, *, threshold=0.0):
        try:
            if value is None:
                return {
                    "available": False,
                    "channel_count": 0,
                    "active_channel_count": 0,
                    "total_positive_pixel_count": 0,
                    "per_channel_positive_pixel_count": [],
                    "threshold": float(threshold),
                }
            if hasattr(value, "detach"):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim != 3:
                return {
                    "available": False,
                    "channel_count": 0,
                    "active_channel_count": 0,
                    "total_positive_pixel_count": 0,
                    "per_channel_positive_pixel_count": [],
                    "threshold": float(threshold),
                }
            mask = arr > float(threshold)
            per_channel = [int(np.count_nonzero(mask[idx])) for idx in range(mask.shape[0])]
            return {
                "available": True,
                "channel_count": int(arr.shape[0]),
                "active_channel_count": int(sum(1 for count in per_channel if count > 0)),
                "total_positive_pixel_count": int(sum(per_channel)),
                "per_channel_positive_pixel_count": per_channel,
                "threshold": float(threshold),
            }
        except Exception as exc:
            return {
                "available": False,
                "error": str(exc),
                "channel_count": 0,
                "active_channel_count": 0,
                "total_positive_pixel_count": 0,
                "per_channel_positive_pixel_count": [],
                "threshold": float(threshold),
            }

    def _refresh_semantic_projection_summary(self, obs=None):
        envs = []
        for env_idx in range(self.num_scenes):
            obs_sem = None
            try:
                if obs is not None and obs.ndim >= 4 and obs.shape[1] > 4:
                    obs_sem = obs[env_idx, 4:, :, :]
            except Exception:
                obs_sem = None
            local_sem = self.local_map[env_idx, 4:, :, :] if self.local_map.shape[1] > 4 else None
            full_sem = self.full_map[env_idx, 4:, :, :] if self.full_map.shape[1] > 4 else None
            instance_footprints = []
            try:
                instance_footprints = list(self.semantic_instance_footprints(env_idx=env_idx) or [])
            except Exception:
                instance_footprints = []
            envs.append(
                {
                    "env_idx": int(env_idx),
                    "obs_semantic_input": self._semantic_channel_counts(obs_sem, threshold=0.0),
                    "local_map_semantic_positive": self._semantic_channel_counts(local_sem, threshold=0.0),
                    "local_map_semantic_thresholded": self._semantic_channel_counts(local_sem, threshold=0.5),
                    "full_map_semantic_positive": self._semantic_channel_counts(full_sem, threshold=0.0),
                    "full_map_semantic_thresholded": self._semantic_channel_counts(full_sem, threshold=0.5),
                    "semantic_instance_footprint_count": int(len(instance_footprints)),
                    "semantic_instance_footprint_pixel_count": int(
                        sum(int(item.get("footprint_pixel_count", 0) or 0) for item in instance_footprints)
                    ),
                    "cat_pred_threshold": float(getattr(self.args, "cat_pred_threshold", 0.0) or 0.0),
                }
            )
        self.last_semantic_projection_summary = {
            "schema_version": "smoothnav.semantic_projection_summary.v1",
            "channel_offset": 4,
            "num_sem_categories": int(self.args.num_sem_categories),
            "envs": envs,
        }
        return self.last_semantic_projection_summary

    def semantic_projection_summary(self, env_idx=0):
        summary = getattr(self, "last_semantic_projection_summary", None) or self._refresh_semantic_projection_summary()
        envs = list(summary.get("envs", []) or [])
        if envs:
            idx = int(env_idx or 0)
            if 0 <= idx < len(envs):
                out = dict(summary)
                out["selected_env"] = dict(envs[idx])
                return out
        return dict(summary)
    
    def mapping(self, obs, infos):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        poses = torch.from_numpy(np.asarray(
            [infos['sensor_pose'] for env_idx
             in range(self.num_scenes)])
        ).float().to(self.device)
        agent_heights = torch.from_numpy(np.asarray(
            [infos['agent_height'] for env_idx in range(self.num_scenes)])
        ).float().to(self.device)

        poses_last = self.local_pose.clone()
        _, self.local_map, _, self.local_pose = \
            self.mapping_module(obs, poses, self.local_map, self.local_pose, agent_heights)
        
        local_pose = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = local_pose + self.origins
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(self.num_scenes):
            # r, c = locs[e, 1], locs[e, 0]
            self.local_row = int(local_pose[e, 1] * 100.0 / self.args.map_resolution)
            self.local_col = int(local_pose[e, 0] * 100.0 / self.args.map_resolution)
            self.local_map[e, 2:4, self.local_row - 2:self.local_row + 3, self.local_col - 2:self.local_col + 3] = 1.
        self._project_semantic_instance_footprints(obs, poses, poses_last, agent_heights, infos)
        self._refresh_semantic_projection_summary(obs=obs)
    
    def move_local_map(self, env_idx=0):
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]
        self.full_pose[env_idx] = self.local_pose[env_idx] + \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

        locs = self.full_pose[env_idx].cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                        self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :,
                                self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1],
                                self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()
        self._refresh_semantic_projection_summary()

    def get_local_map_boundaries(self, agent_loc):
        loc_r, loc_c = agent_loc

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - self.local_width // 2, loc_c - self.local_height // 2
            gx2, gy2 = gx1 + self.local_width, gy1 + self.local_height
            if gx1 < 0:
                gx1, gx2 = 0, self.local_width
            if gx2 > self.global_width:
                gx1, gx2 = self.global_width - self.local_width, self.global_width

            if gy1 < 0:
                gy1, gy2 = 0, self.local_height
            if gy2 > self.global_height:
                gy1, gy2 = self.global_height - self.local_height, self.global_height
        else:
            gx1, gx2, gy1, gy2 = 0, self.global_width, 0, self.global_height

        return [gx1, gx2, gy1, gy2]
    
    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.last_semantic_instance_footprints = {
            "schema_version": "smoothnav.semantic_instance_footprints.v1",
            "envs": [],
        }
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.local_map_boundary[e] = self.get_local_map_boundaries((loc_r, loc_c))

            self.planner_pose_inputs[e, 3:] = self.local_map_boundary[e]
            self.origins[e] = [self.local_map_boundary[e][2] * self.args.map_resolution / 100.0,
                          self.local_map_boundary[e][0] * self.args.map_resolution / 100.0, 0.]

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :,
                                    self.local_map_boundary[e, 0]:self.local_map_boundary[e, 1],
                                    self.local_map_boundary[e, 2]:self.local_map_boundary[e, 3]]
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()
        self._refresh_semantic_projection_summary()

    def init_map_and_pose_for_env(self, env_idx=0):
        self.full_map[env_idx].fill_(0.)
        self.full_pose[env_idx].fill_(0.)
        envs = list((getattr(self, "last_semantic_instance_footprints", {}) or {}).get("envs", []) or [])
        if 0 <= int(env_idx) < len(envs):
            envs[int(env_idx)] = {"env_idx": int(env_idx), "input_instance_count": 0, "projected_instance_count": 0, "instances": [], "warnings": []}
            self.last_semantic_instance_footprints = {
                "schema_version": "smoothnav.semantic_instance_footprints.v1",
                "envs": envs,
            }
        self.full_pose[env_idx, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose[env_idx].cpu().numpy()
        self.planner_pose_inputs[env_idx, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[env_idx, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                      self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()
        self._refresh_semantic_projection_summary()

    def update_intrinsic_rew(self, env_idx=0):
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]
