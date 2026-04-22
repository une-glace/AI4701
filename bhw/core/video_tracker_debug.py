from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def init_map_stats() -> dict[str, int]:
    """Initialize counters used to inspect map-dedup and motion-estimation behavior."""
    return {
        "motion_attempts": 0,
        "motion_successes": 0,
        "motion_failures": 0,
        "phase_fallback_successes": 0,
        "phase_fallback_failures": 0,
        "disabled_frames": 0,
        "center_unavailable": 0,
        "dedup_checks": 0,
        "dedup_hits": 0,
    }


def write_summary_file(
    summary_path: Path,
    source: Path,
    weights: Path,
    frames: int,
    unique_tracks: int,
    count_by_class: Mapping[str, int],
    map_debug_summary: bool,
    map_stats: Mapping[str, int],
) -> None:
    """Persist one run summary for downstream scripts and reproducible debugging."""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"source: {source}\n")
        f.write(f"weights: {weights}\n")
        f.write(f"frames: {frames}\n")
        f.write(f"unique_tracks: {unique_tracks}\n")
        if map_debug_summary:
            f.write("\n# Map Summary\n")
            f.write(f"motion_attempts: {map_stats['motion_attempts']}\n")
            f.write(f"motion_successes: {map_stats['motion_successes']}\n")
            f.write(f"motion_failures: {map_stats['motion_failures']}\n")
            f.write(f"phase_fallback_successes: {map_stats['phase_fallback_successes']}\n")
            f.write(f"phase_fallback_failures: {map_stats['phase_fallback_failures']}\n")
            f.write(f"dedup_checks: {map_stats['dedup_checks']}\n")
            f.write(f"dedup_hits: {map_stats['dedup_hits']}\n")
            f.write(f"center_unavailable: {map_stats['center_unavailable']}\n")
            f.write(f"map_disabled_events: {map_stats['disabled_frames']}\n")
        f.write("\n# Counts (by class)\n")
        for class_name, count in sorted(count_by_class.items(), key=lambda x: x[0]):
            f.write(f"{class_name}: {count}\n")


def print_runtime_report(
    frame_idx: int,
    inferred_frames: int,
    target_infer_fps: float,
    infer_stride: int,
    effective_conf: float,
    base_conf: float,
    stride_aware_conf: bool,
    effective_min_track_frames: int,
    min_track_seconds: float,
    history_maxlen: int,
    dedup_stride_scale: float,
    effective_recount_center_ratio: float,
    effective_map_center_ratio: float,
    split_sep_iou_max: float,
    split_sep_dist_scale: float,
    split_sep_local_diag_scale: float,
    split_sep_min_dist_px: float,
    map_debug_summary: bool,
    map_stats: Mapping[str, int],
    effective_track_buffer: int | None,
    track_buffer_seconds: float | None,
    grabbed_skip_frames: int,
    use_fast_grab_skip: bool,
    unique_tracks: int,
    count_by_class: Mapping[str, int],
    saved_vis_count: int,
    summary_path: Path,
    annotated_video_path: Path | None,
    elapsed: float,
) -> None:
    """Print a concise post-run report for CLI debugging and benchmark tracking."""
    print("\n=== Run Summary ===")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames inferred: {inferred_frames} (target infer_fps={target_infer_fps:.3f}, stride={infer_stride})")
    print(
        f"Effective conf: {effective_conf:.3f} (base_conf={base_conf:.3f}, stride_aware_conf={stride_aware_conf})"
    )
    print(
        f"Effective min_track_frames: {effective_min_track_frames} "
        f"(min_track_seconds={min_track_seconds:.3f}, history_maxlen={history_maxlen})"
    )
    print(
        f"Dedup stride scale: {dedup_stride_scale:.3f} "
        f"(recount_center_ratio={effective_recount_center_ratio:.5f}, map_center_ratio={effective_map_center_ratio:.5f})"
    )
    print(
        "Split tuning: "
        f"iou_max={split_sep_iou_max:.3f}, "
        f"dist_scale={split_sep_dist_scale:.3f}, "
        f"local_diag_scale={split_sep_local_diag_scale:.3f}, "
        f"min_dist_px={split_sep_min_dist_px:.1f}"
    )

    if map_debug_summary:
        dedup_hit_rate = (
            float(map_stats["dedup_hits"]) / float(map_stats["dedup_checks"])
            if map_stats["dedup_checks"] > 0
            else 0.0
        )
        motion_success_rate = (
            float(map_stats["motion_successes"]) / float(map_stats["motion_attempts"])
            if map_stats["motion_attempts"] > 0
            else 0.0
        )
        phase_total = map_stats["phase_fallback_successes"] + map_stats["phase_fallback_failures"]
        print(
            "Map summary: "
            f"motion_success={map_stats['motion_successes']}/{map_stats['motion_attempts']} ({motion_success_rate:.1%}), "
            f"phase_fallback={map_stats['phase_fallback_successes']}/{phase_total}, "
            f"dedup_hits={map_stats['dedup_hits']}/{map_stats['dedup_checks']} ({dedup_hit_rate:.1%}), "
            f"center_unavailable={map_stats['center_unavailable']}, "
            f"map_disabled_events={map_stats['disabled_frames']}"
        )

    print("\n=== Counts By Class ===")
    if count_by_class:
        for class_name, count in sorted(count_by_class.items(), key=lambda x: x[0]):
            print(f"{class_name}: {count}")
    else:
        print("No counted classes.")

    if effective_track_buffer is not None and track_buffer_seconds is not None:
        print(
            f"Effective track_buffer: {effective_track_buffer} frames "
            f"(~{float(track_buffer_seconds):.2f}s at infer_fps={target_infer_fps:.2f})"
        )

    print("\n=== Artifacts ===")
    print(f"Frames skipped by grab: {grabbed_skip_frames} (fast_grab_skip={use_fast_grab_skip})")
    print(f"Unique tracks: {unique_tracks}")
    print(f"Saved visualizations: {saved_vis_count}")
    print(f"Summary saved to: {summary_path}")
    if annotated_video_path is not None:
        print(f"Annotated video saved to: {annotated_video_path}")

    print("\n=== Timing ===")
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"Processing time: {h:02d}:{m:02d}:{s:02d} ({elapsed:.2f}s)")
    if elapsed > 0:
        print(f"Average processing FPS: {frame_idx / elapsed:.2f}")
