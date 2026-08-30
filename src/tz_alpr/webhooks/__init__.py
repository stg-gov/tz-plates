"""Webhook delivery (spec §20). Implemented in Phase 4.

Planned contract, frozen now so consumers can build against it:
  * event: "vehicle.detected"
  * fields: event_id (idempotency key), camera_id, timestamp, plate, confidence,
    vehicle_type
  * HMAC-SHA256 signature header over the raw body using TZ_ALPR_WEBHOOK_SECRET
  * at-least-once delivery: exponential-backoff retries, per-attempt timeout,
    receiver deduplicates on event_id
"""
