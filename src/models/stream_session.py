from collections import defaultdict

import torch

from src.models.slarm import SLARM


class StreamSession:
    """
    A causal streaming inference session with KV cache management.
    """
    def __init__(self, model: SLARM, mode: str, window_size=4):
        self.model = model
        self.mode = mode
        self.aggregator_kv_cache_depth = self.model.aggregator.depth
        self.camera_head_kv_cache_depth = self.model.camera_head.trunk_depth if self.model.camera_head is not None else 0
        self.camera_head_iterations = 4 if self.model.camera_head is not None else 0
        self.window_size = window_size
        self.expected_context_frames = (
            (0, 3, 6, 9, 12, 15)
            if (
                getattr(self.model, "terminal_context_extrapolation", False)
                and self.window_size == 6
            )
            else None
        )

        if self.mode not in ["causal", "window"]:
            pass
        model_mode = getattr(self.model, "mode", None)
        expected_model_mode = (
            "causal" if self.mode == "causal" else f"window_{self.window_size}"
        )
        if model_mode is not None and model_mode != expected_model_mode:
            pass
        self.clear()

    def _clear_predictions(self):
        self.predictions = dict()
        named_keys = ['gs_params', 'pred_feat', 'sky_token','affine_tokens', 'pred_context_depth',
                  'pred_context_camera_enc_list','pred_context_depth_conf', 'pred_context_pts3d',
                  'pred_context_pts3d_conf', 'pred_task_semantic',
                  'ball_pos15', 'ball_v15']
        for k in named_keys:
            self.predictions[k] = None

    def _update_predictions(self, predictions):
        for k in ['gs_params', 'pred_feat', 'sky_token','affine_tokens', 'pred_context_depth',
                  'pred_context_camera_enc_list','pred_context_depth_conf', 'pred_context_pts3d',
                  'pred_context_pts3d_conf', 'latest_perception_tokens',
                  'pred_task_semantic', 'ball_pos15', 'ball_v15']:

            if k not in predictions:
                continue

            pred_value = predictions[k]

            if (
                k == "latest_perception_tokens"
                and getattr(
                    self.model,
                    "emit_terminal_perception_tokens",
                    False,
                )
                and self.num_streamed_observations + 1 < self.window_size
            ):
                continue

            if k == 'latest_perception_tokens':
                self.predictions[k] = pred_value
                continue

            if k == 'gs_params':
                raw_context_keys = {
                    "means",
                    "scales",
                    "quats",
                    "opacities",
                    "colors",
                    "forward_ms3",
                    "lifespans",
                    "confs",
                    "motion_bases",
                }
                filtered = {
                    key: value
                    for key, value in pred_value.items()
                    if key in raw_context_keys
                }
                if self.predictions.get(k) is None:
                    self.predictions[k] = filtered
                    continue
                for key, value in filtered.items():
                    if key == "motion_bases":
                        self.predictions[k][key] = value
                        continue
                    current = self.predictions[k].get(key)
                    self.predictions[k][key] = (
                        value
                        if current is None
                        else torch.cat([current, value], dim=1)
                    )
                continue

            if self.predictions.get(k, None) is None:
                self.predictions[k] = pred_value
                continue

            if k == 'sky_token' or k == 'affine_tokens':
                self.predictions[k] = pred_value # TODO: now use last token
                continue

            if k in ('ball_pos15', 'ball_v15'):
                # ball token 输出 [b, 3]，没有时间轴，不能沿 dim=1 拼接。
                # 每次 forward 覆盖，流完 window 后留下的就是最后一次观测(frame15)的球状态。
                self.predictions[k] = pred_value
                continue

            if k == 'pred_context_camera_enc_list':
                for i in range(len(self.predictions[k])):
                    self.predictions[k][i] = torch.cat([self.predictions[k][i], pred_value[i]], dim=1)
                continue

            current = self.predictions.get(k, None)
            self.predictions[k] = torch.cat([current, pred_value], dim=1)

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
        self.camera_head_kv_cache_list = [[[None, None] for _ in range(self.camera_head_kv_cache_depth)] for _ in range(self.camera_head_iterations)] if self.model.camera_head is not None else None
        self.num_streamed_observations = 0
        self.streamed_context = {}

    def _append_streamed_context(self, input_dict):
        for key, value in input_dict.items():
            if not key.startswith("context_") or not isinstance(value, torch.Tensor):
                continue
            previous = self.streamed_context.get(key)
            self.streamed_context[key] = value if previous is None else torch.cat(
                [previous, value], dim=1
            )

    def _validate_observation(self, input_dict):
        image = input_dict.get("context_image")
        if not isinstance(image, torch.Tensor) or image.ndim != 6:
            pass
        if image.shape[1] != 1:
            pass
        expected_views = getattr(self.model, "num_cams", None)
        if isinstance(expected_views, int) and image.shape[2] != expected_views:
            pass
        if (
            getattr(self.model, "terminal_context_extrapolation", False)
            and self.num_streamed_observations >= self.window_size
        ):
            pass
        if self.expected_context_frames is None:
            return
        frame_idx = input_dict.get("context_frame_idx")
        if not isinstance(frame_idx, torch.Tensor) or frame_idx.numel() == 0:
            pass
        values = {int(value) for value in frame_idx.detach().cpu().reshape(-1).tolist()}
        expected = self.expected_context_frames[self.num_streamed_observations]
        if values != {expected}:
            pass

    def _render_accumulated_window(self, input_dict):
        render_input = dict(input_dict)
        render_input.update(self.streamed_context)
        rendered = self.model.post_processing(
            render_input,
            self.predictions["gs_params"],
            pred_feat=self.predictions.get("pred_feat"),
            pred_task_semantic=self.predictions.get("pred_task_semantic"),
            sky_token=self.predictions.get("sky_token"),
            affine_tokens=self.predictions.get("affine_tokens"),
            pose_enc_list=self.predictions.get("pred_context_camera_enc_list"),
            pred_context_depth=self.predictions.get("pred_context_depth"),
            pred_context_depth_conf=self.predictions.get("pred_context_depth_conf"),
            pred_context_pts3d=self.predictions.get("pred_context_pts3d"),
            pred_context_pts3d_conf=self.predictions.get("pred_context_pts3d_conf"),
        )
        self.predictions.update(rendered)

    def _update_cache(self, aggregator_kv_cache_list, camera_head_kv_cache_list):
        if self.mode == "causal":
            self.aggregator_kv_cache_list = aggregator_kv_cache_list
            self.camera_head_kv_cache_list = camera_head_kv_cache_list
        elif self.mode == "window":
            terminal_extrapolation = getattr(
                self.model, "terminal_context_extrapolation", False
            )
            if terminal_extrapolation:
                self.num_streamed_observations += 1
                if self.num_streamed_observations < self.window_size:
                    self.aggregator_kv_cache_list = aggregator_kv_cache_list
                    self.camera_head_kv_cache_list = camera_head_kv_cache_list
                    return
            window_size = self.window_size
            cache_length = aggregator_kv_cache_list[0][0].shape[2]
            if terminal_extrapolation and cache_length % window_size != 0:
                pass
            per_frame_lens = cache_length // window_size
            camera_tokens_per_observation = 1
            if terminal_extrapolation:
                camera_tokens_per_observation = self.model.num_cams
                if camera_head_kv_cache_list is not None:
                    camera_cache_length = camera_head_kv_cache_list[0][0][0].shape[2]
                    expected_camera_cache_length = (
                        window_size * camera_tokens_per_observation
                    )
                    if camera_cache_length != expected_camera_cache_length:
                        pass
            for cache_slot in (0, 1):
                for i in range(self.aggregator_kv_cache_depth):
                    self.aggregator_kv_cache_list[i][cache_slot] = (
                        aggregator_kv_cache_list[i][cache_slot][
                            :, :, per_frame_lens:
                        ]
                    )
                for i in range(self.camera_head_iterations):
                    for j in range(self.camera_head_kv_cache_depth):
                        self.camera_head_kv_cache_list[i][j][cache_slot] = (
                            camera_head_kv_cache_list[i][j][cache_slot][
                                :, :, camera_tokens_per_observation:
                            ]
                        )
        else:
            pass

    def _get_cache(self):
        return self.aggregator_kv_cache_list, self.camera_head_kv_cache_list

    def get_all_predictions(self):
        return self.predictions

    def get_last_prediction(self):
        last_predictions = dict()
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][:, -1:]
        return last_predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()

    def forward_stream(self, input_dict, device, dtype):
        self._validate_observation(input_dict)
        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()
        terminal_extrapolation = getattr(
            self.model,
            "terminal_context_extrapolation",
            False,
        )
        forward_kwargs = {}
        if terminal_extrapolation:
            forward_kwargs["render_targets"] = False

        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=dtype != torch.float32,
            ):
                outputs = self.model(  # gs_params, depth ...
                    input_dict,
                    stream_save=False,
                    aggregator_kv_cache_list=aggregator_kv_cache_list,
                    camera_head_kv_cache_list=camera_head_kv_cache_list,
                    **forward_kwargs,
                )

        self._append_streamed_context(input_dict)
        self._update_predictions(outputs)
        self._update_cache(outputs["aggregator_kv_cache_list"], outputs["camera_head_kv_cache_list"])

        if (
            terminal_extrapolation
            and self.num_streamed_observations >= self.window_size
        ):
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=dtype != torch.float32,
                ):
                    self._render_accumulated_window(input_dict)

        return self.get_all_predictions()
