---
title: tz-alpr Tanzania Plates
emoji: 🚘
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12"
short_description: Read Tanzanian car and bajaji plates
startup_duration_timeout: 30m
---

# tz-alpr

Upload a vehicle-rear photo. The Space detects the plate, runs the CRNN+CTC
OCR checkpoint, and applies Tanzania plate rules (`T###ABC` / `MC###ABC`).

Model: linked from this Space’s sibling repo after upload.
