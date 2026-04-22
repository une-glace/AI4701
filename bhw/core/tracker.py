#!/usr/bin/env python3
from __future__ import annotations

try:
    from core.video_tracker import parse_args, process_video
except ImportError:
    from video_tracker import parse_args, process_video


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
