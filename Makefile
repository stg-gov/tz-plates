.PHONY: help install install-train test lint api synth prepare train-ocr pretrain-ocr \
        train-detector train-vehicle-detector evaluate export bench docker video

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install:            ## install runtime (inference API) deps
	pip install -e .

install-train:      ## install training + detector + onnx extras
	pip install -e ".[train,detect,onnx,dev]"

test:              ## run the test suite
	pytest -q

lint:              ## ruff check
	ruff check src tools training tests

api:               ## run the API locally with reload
	TZ_ALPR_ENV=development uvicorn tz_alpr.api.main:app --reload --port 8080

synth:             ## generate synthetic Tanzanian plates for OCR pretraining
	python tools/generate_tanzania_plates.py --count 40000

prepare:           ## build OCR crops + splits from labeled_images/ + labels.jsonl
	python tools/prepare_dataset.py --workers 8

pretrain-ocr:      ## stage 1: pretrain OCR on synthetic only
	python training/train_ocr.py --stage pretrain

train-ocr:         ## stage 2: fine-tune OCR on real crops
	python training/train_ocr.py --stage finetune --init models/ocr/v1/ocr_crnn_pretrained.pt

train-detector:    ## train the YOLO plate detector (needs verified boxes)
	python training/train_plate_detector.py

train-vehicle-detector:  ## fine-tune the vehicle detector for minibus / tuk-tuk
	python training/train_vehicle_detector.py

evaluate:          ## evaluate + calibrate on the test split
	python training/evaluate.py --split test --calibrate
	python training/evaluate.py --split hard_test

export:            ## export the trained OCR model to ONNX
	python tools/export_onnx.py ocr --ckpt checkpoints/last.ckpt

bench:             ## benchmark the pipeline
	python tools/benchmark.py --limit 200

video:             ## run the video pipeline on a clip: make video VID=clip.mp4 CAM=CAM_01
	python tools/process_video.py $(VID) --camera-id $(or $(CAM),upload) --sample-fps 5

docker:            ## build the runtime image
	docker build --target runtime -t tz-alpr:latest .
