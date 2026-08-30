# Build a Production-Grade Tanzanian Automatic License Plate Recognition System

Act as a senior Computer Vision / Machine Learning Engineer specializing in:

* Automatic License Plate Recognition (ALPR/ANPR)
* Object detection
* OCR
* Video analytics
* PyTorch
* PyTorch Lightning
* OpenCV
* YOLO
* RTSP video processing
* ONNX
* TensorRT
* Docker
* FastAPI
* Edge AI deployment

I want you to design and implement an independent, open-source Automatic License Plate Recognition platform with capabilities comparable to commercial ALPR platforms such as Plate Recognizer.

Do not copy proprietary source code, models, weights, APIs, or other protected implementation details.

Build the system from open-source technologies and models.

The initial target country is:

**Tanzania**

The architecture must later allow adding:

* Kenya
* Uganda
* Rwanda
* Burundi
* Zambia
* Malawi
* Mozambique
* other African countries

The first production use case will be parking management and road-reserve parking enforcement.

---

# 1. Main Objective

Build an ALPR system capable of processing:

1. Single images
2. Uploaded videos
3. RTSP camera streams
4. CCTV cameras
5. Mobile-phone/POS camera images

The system should detect vehicles, locate number plates, recognize plate characters, normalize Tanzanian registration numbers, estimate confidence, and return structured JSON.

Target workflow:

Camera/Image
->
Image Enhancement
->
Vehicle Detection
->
License Plate Detection
->
Plate Alignment / Rectification
->
OCR
->
Tanzania Plate Validation
->
Temporal Validation
->
Vehicle Tracking
->
Structured Result
->
REST API / Webhook

---

# 2. Tanzanian Number Plates

The system must initially be optimized for Tanzanian vehicle registration plates.

Examples include patterns similar to:

T 331 EBG

T 123 ABC

T 456 DEF

The system must account for different Tanzanian vehicle categories and plate layouts.

Create a configurable plate-pattern engine rather than hard-coding everything into the neural network.

Example normalized result:

T331EBG

while preserving:

raw_text: "T 331 EBG"
normalized_text: "T331EBG"

Create:

src/country_rules/tanzania.py

This module should contain configurable:

* regex rules
* formatting rules
* character-position rules
* plate categories
* confidence adjustments
* common OCR corrections

Do not blindly convert characters.

For example, ambiguity may occur between:

0/O
1/I
2/Z
5/S
6/G
8/B

Use expected plate structure and OCR probabilities to determine the most likely character.

---

# 3. Architecture

Use a modular multi-stage architecture.

## Stage 1 — Vehicle Detection

Detect:

* car
* motorcycle
* bus
* truck
* minibus
* tuk-tuk / bajaji
* other configurable vehicle classes

Use a modern YOLO-family detector or another strong open-source detector.

Make the detector replaceable.

Example interface:

VehicleDetector.detect(image)

Return:

bounding box
vehicle class
confidence

---

# 4. License Plate Detection

Create a dedicated plate detector.

Do not depend only on the generic vehicle detector.

Pipeline:

image
->
vehicle detector
->
vehicle crop
->
plate detector
->
plate bounding box

Support multiple vehicles and multiple plates in the same image.

Return plate coordinates relative to both:

* vehicle
* original image

---

# 5. Plate Rectification

Real-world images will contain plates that are:

* tilted
* rotated
* photographed from the side
* perspective distorted
* partially blurred

Implement plate rectification.

Investigate:

* corner detection
* keypoint detection
* homography
* perspective transformation
* Spatial Transformer Networks

Pipeline:

detected plate
->
corner/keypoint detection
->
perspective correction
->
normalized plate image

Create:

src/rectification/

---

# 6. OCR Model

The OCR component is extremely important.

Compare suitable approaches including:

* CRNN + CTC
* LPRNet
* SVTR
* PARSeq
* TrOCR
* transformer-based OCR
* lightweight CNN-transformer architectures

Select an architecture appropriate for license plates.

Explain why it was selected.

The OCR model must output:

characters
character probabilities
sequence confidence

Example:

{
"text": "T331EBG",
"confidence": 0.96
}

Do not rely solely on generic Tesseract OCR.

---

# 7. Tanzania-Aware Recognition

Create a post-processing engine that combines:

OCR probabilities

*

Tanzanian plate rules

*

known character positions

*

temporal observations

to generate the final prediction.

Example:

OCR:

T33IEBG

But if the expected position requires a numeric character, determine whether:

I -> 1

is appropriate based on OCR probabilities.

Return both predictions:

{
"raw_ocr": "T33IEBG",
"normalized_plate": "T331EBG"
}

Never hide corrections.

---

# 8. Difficult Conditions

The model must be designed to handle:

* nighttime images
* low-resolution CCTV
* motion blur
* dirty plates
* faded plates
* shadows
* glare
* rain
* strong sunlight
* angled vehicles
* partially occluded plates
* motorcycles
* two-line plates
* multiple vehicles
* distant vehicles

Create augmentation pipelines representing these conditions.

Use Albumentations where appropriate.

---

# 9. Super-Resolution

Evaluate whether plate super-resolution improves OCR.

Pipeline:

small plate
->
super-resolution
->
OCR

Possible approaches:

Real-ESRGAN
or
lightweight custom SR model

Super-resolution must be optional because it increases inference latency.

---

# 10. Dataset Architecture

Design the dataset as:

datasets/
raw/
images/
videos/

```
annotations/
    vehicles/
    plates/
    ocr/

processed/
    vehicle_crops/
    plate_crops/
    rectified_plates/

synthetic/
    generated_plates/
```

Create dataset preparation scripts.

---

# 11. Annotation Format

Use YOLO format for detection where appropriate.

For OCR create:

ocr_annotations.csv

Example:

image_path,plate_text,country,plate_type

plates/000001.jpg,T331EBG,TZ,PRIVATE

Also create a unified metadata format containing:

image_id
camera_id
timestamp
vehicle_bbox
plate_bbox
plate_text
plate_type
vehicle_type
weather
lighting
angle
occlusion
blur_score

---

# 12. Synthetic Tanzanian Plates

Real data may initially be limited.

Build a synthetic plate generator.

Create:

tools/generate_tanzania_plates.py

Generate realistic Tanzanian-style plates with configurable:

* fonts
* spacing
* backgrounds
* character combinations
* plate sizes
* perspective
* brightness
* shadows
* blur
* noise
* compression
* dirt
* scratches
* glare
* day/night appearance

Synthetic images should be used for pretraining.

Then fine-tune using real Tanzanian plate images.

---

# 13. Dataset Splitting

Avoid data leakage.

Do not randomly split frames from the same video across training and validation.

Split by:

camera
location
video
capture session

where possible.

Create:

train
validation
test
hard_test

The hard test set should include:

night
blur
rain
angles
motorcycles
small plates
partial occlusion

---

# 14. Training Framework

Use:

Python
PyTorch
PyTorch Lightning

Structure training using LightningModules and LightningDataModules.

Support:

GPU
mixed precision
checkpointing
early stopping
TensorBoard
Weights & Biases optional

Configuration should be YAML based.

Example:

configs/
detector.yaml
plate_detector.yaml
ocr.yaml
training.yaml
tanzania.yaml

---

# 15. Evaluation Metrics

Do not report only OCR character accuracy.

Measure:

Plate Detection:

Precision
Recall
mAP@50
mAP@50:95

OCR:

Character Accuracy
Character Error Rate
Sequence Accuracy

End-to-End ALPR:

Exact Plate Match Accuracy

Example:

ground truth:

T331EBG

prediction:

T331EBG

Only an exact match counts as a successful plate recognition.

Also report:

day accuracy
night accuracy
motorcycle accuracy
car accuracy
front/rear
camera-specific accuracy

---

# 16. Confidence Scoring

Generate several confidence values:

vehicle_confidence
plate_detection_confidence
ocr_confidence
plate_validation_confidence
final_confidence

Do not simply multiply all scores.

Design and document a calibrated confidence strategy.

---

# 17. Temporal Recognition

For video, do not trust one frame.

Track vehicles using:

ByteTrack

or another appropriate open-source tracker.

Architecture:

YOLO
->
ByteTrack
->
plate detector
->
OCR
->
temporal aggregation

For one vehicle track:

Frame 1: T331E8G 0.72
Frame 2: T331EBG 0.91
Frame 3: T331EBG 0.96
Frame 4: T331E8G 0.80

Final:

T331EBG
confidence: 0.97

Implement weighted temporal voting using OCR probabilities.

---

# 18. Duplicate Prevention

Parking cameras may observe the same vehicle for several seconds.

Do not generate a new event for every frame.

Create event deduplication based on:

tracking ID
plate
camera
timestamp
location

Example:

same vehicle detected across 50 frames

should produce:

ONE vehicle event

not 50 events.

---

# 19. REST API

Build the inference API using FastAPI.

Required endpoint:

POST /v1/plate-reader

Accept:

multipart image upload

Return JSON.

Example:

{
"processing_time_ms": 83,
"results": [
{
"plate": "T331EBG",
"raw_text": "T 331 EBG",
"confidence": 0.97,

```
        "plate_bbox": {
            "x": 420,
            "y": 320,
            "width": 180,
            "height": 55
        },

        "vehicle": {
            "type": "car",
            "confidence": 0.95
        },

        "country": "TZ",
        "plate_type": "PRIVATE"
    }
]
```

}

Additional endpoints:

GET /health

GET /version

POST /v1/video

POST /v1/streams

DELETE /v1/streams/{id}

GET /v1/streams/{id}

---

# 20. Webhooks

Implement webhook support.

When a plate is recognized:

POST configured webhook URL.

Example payload:

{
"event": "vehicle.detected",
"camera_id": "DODOMA_PARKING_01",
"timestamp": "...",
"plate": "T331EBG",
"confidence": 0.97,
"vehicle_type": "car"
}

Implement:

retry
timeouts
signatures
event IDs
idempotency

---

# 21. RTSP Streaming

Implement:

RTSP camera
->
frame reader
->
frame sampling
->
vehicle detector
->
tracker
->
plate detector
->
OCR
->
temporal aggregation
->
event
->
webhook

The system must support multiple cameras.

Configuration:

cameras:

* id: dodoma_001
  url: rtsp://...
  fps: 5

* id: dodoma_002
  url: rtsp://...
  fps: 5

Credentials must come from environment variables or secrets, not source-controlled YAML.

---

# 22. Parking Enforcement Workflow

Design the system for integration with a Tanzanian parking-management platform.

Workflow:

Parking officer / camera
->
capture vehicle
->
detect number plate
->
validate registration
->
send plate to parking system
->
identify parking zone
->
calculate applicable charge
->
generate parking bill
->
return bill/reference
->
support payment through external payment system

Keep ALPR and billing as separate services.

The ALPR system should produce recognition events.

The parking system should handle billing.

---

# 23. Human Verification

Low-confidence recognition must support manual verification.

Example:

confidence >= 0.90
automatic acceptance

0.70 - 0.89
configurable review

< 0.70
manual verification / rejection

Thresholds must be configurable and eventually calibrated from validation data.

Create a simple review interface where an operator can see:

vehicle image
plate crop
recognized plate
confidence
alternative OCR candidates

and approve or correct the result.

Corrections should be stored for future model retraining.

---

# 24. Active Learning

Create an active-learning pipeline.

Automatically identify:

low confidence plates
OCR disagreement
unusual plate patterns
new plate types
difficult lighting
incorrect operator-verified predictions

Send these examples to:

datasets/retraining_queue/

This should create a continuous improvement loop:

Production
->
difficult samples
->
human verification
->
annotation
->
retraining
->
evaluation
->
model registry
->
deployment

---

# 25. Privacy

Do not store full vehicle images indefinitely unless required by the consuming application.

Support configurable:

image retention
plate crop retention
face blurring
plate blurring for exported/public imagery
audit logging

Separate inference from long-term evidence storage.

---

# 26. Deployment

Everything must run on-premise.

Create Docker containers.

Services:

alpr-api
stream-worker
optional-review-ui
redis
postgres

Use Docker Compose for development.

Prepare architecture for Kubernetes later.

Target hardware:

CPU server

NVIDIA GPU server

Jetson edge devices

Optimize using:

ONNX Runtime

and optionally:

TensorRT

---

# 27. Repository Structure

Create a professional repository:

tanzania-alpr/

```
README.md

pyproject.toml

requirements.txt

docker-compose.yml

Dockerfile

configs/

datasets/

models/

checkpoints/

src/

    detection/

    plate_detection/

    rectification/

    ocr/

    tracking/

    country_rules/

    preprocessing/

    postprocessing/

    api/

    streaming/

    webhooks/

    utils/

training/

    train_plate_detector.py

    train_ocr.py

    evaluate.py

tools/

    prepare_dataset.py

    extract_frames.py

    generate_tanzania_plates.py

    export_onnx.py

    benchmark.py

tests/

notebooks/
```

---

# 28. Testing

Create automated tests for:

plate normalization
Tanzania regex validation
OCR post-processing
API
webhook generation
duplicate suppression
tracking
confidence thresholds

Include unit and integration tests.

---

# 29. Benchmarking

Create:

tools/benchmark.py

Report:

FPS
average latency
P50 latency
P95 latency
CPU utilization
GPU utilization
RAM
VRAM

Benchmark separately:

vehicle detection
plate detection
OCR
complete pipeline

---

# 30. Model Versioning

Every response should include model information.

Example:

{
"model_version": "tz-alpr-1.0.0"
}

Support model registry structure:

models/
detector/
v1/
v2/

```
ocr/
    v1/
    v2/
```

Do not silently replace production models.

---

# 31. MVP

Do not attempt every feature initially.

Build MVP in this order:

Phase 1

Image
->
Plate Detector
->
OCR
->
Tanzania Validation
->
JSON

Phase 2

Vehicle Detection
->
Plate Detection
->
OCR

Phase 3

Video
->
Tracking
->
Temporal OCR

Phase 4

RTSP cameras + webhooks

Phase 5

Parking-system integration

Phase 6

Vehicle attributes and advanced analytics

---

# 32. Vehicle Attributes — Later Phase

Design extension points for eventually detecting:

vehicle type
make
model
color
orientation
direction of travel

Do not make these features block the initial plate-recognition MVP.

---

# 33. Performance Goals

Treat these as engineering targets, not guaranteed results.

Target end-to-end exact-match accuracy:

> = 95% under normal Tanzanian conditions

Target strong performance for:

daylight
night
motion blur
motorcycles
angled plates

Target GPU latency:

< 100 ms/image where hardware permits.

Target CPU deployment should remain usable without requiring a GPU.

Measure all claims experimentally.

---

# 34. Deliverables

I do not want only an explanation.

Generate the actual project.

Start by providing:

1. Architecture
2. Technology choices
3. Repository structure
4. Dataset specification
5. Annotation strategy
6. Model architecture
7. Tanzania plate-rule engine
8. Dataset preprocessing code
9. Synthetic plate generator
10. Plate detector training code
11. OCR training code
12. PyTorch Lightning modules
13. Evaluation code
14. Inference pipeline
15. FastAPI service
16. RTSP processing
17. ByteTrack integration
18. Temporal OCR aggregation
19. Webhook implementation
20. Docker deployment
21. Tests
22. README
23. Example API calls

For every file:

Show its full path followed by the complete implementation.

Do not provide pseudocode when executable code can reasonably be provided.

Do not leave placeholders such as:

TODO
implement later
your code here

unless an external credential, dataset path, or deployment-specific value genuinely must be supplied by me.

---

# 35. Development Strategy

Do not generate the entire system blindly in one response.

Work iteratively.

First create:

PHASE 1 — ALPR MVP

The MVP must accept:

POST /v1/plate-reader

with a Tanzanian vehicle image and return:

{
"plate": "T331EBG",
"confidence": 0.95
}

Implement the complete Phase 1 repository first.

After completing it:

explain how to train the detector and OCR model using my Tanzanian dataset.

Then provide the commands to:

prepare dataset
train
evaluate
export model
start API
test image

The project should be designed so that later I can say:

"Proceed to Phase 2"

and you can extend the same repository without rewriting the architecture.
