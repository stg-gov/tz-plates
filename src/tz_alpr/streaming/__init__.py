"""RTSP / video stream workers (spec §21). Implemented in Phase 3-4.

The Phase 1 architecture already fixes the shape: a stream worker will pull
frames, sample to the configured FPS, run ``AlprPipeline`` per sampled frame,
push detections through the tracker + temporal aggregator, deduplicate into
events, and emit webhooks. Camera credentials come from the environment, never
from source-controlled YAML.
"""
