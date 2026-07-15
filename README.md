# Logo Detection App

FastAPI backend and static frontend for a logo detection MLOps workflow. The app manages projects, uploaded images and videos, unique frame extraction, CVAT annotation handoff, YOLO label import, approved dataset creation, training runs, model registry, and detection runs.

This README documents the implemented application, including the current Celery training and detection workers, model registry, static frontend, local artifact storage, and PostgreSQL metadata layer.

## Table of Contents

- [Current Scope](#current-scope)
- [Architecture Overview](#architecture-overview)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Step-by-Step Local Setup](#step-by-step-local-setup)
- [Step-by-Step Product Workflow](#step-by-step-product-workflow)
- [API Reference](#api-reference)
- [Storage Layout](#storage-layout)
- [File Formats](#file-formats)
- [Database Architecture](#database-architecture)
- [Configuration](#configuration)
- [Status Model](#status-model)
- [Frontend](#frontend)
- [Target Production Architecture](#target-production-architecture)
- [Operational Notes](#operational-notes)

## Current Scope

Implemented now:

- Project CRUD.
- Class registry CRUD for stable YOLO class IDs.
- Multi-file upload for images and videos.
- Local file storage under `storage/`.
- OpenCV-based video frame extraction.
- Blur filtering by Laplacian variance.
- Duplicate frame filtering by perceptual hash.
- Annotation item listing and image preview.
- CVAT task creation and upload through `app/services/cvat_client.py`.
- Optional CVAT auto-annotation and label sync actions for created tasks.
- YOLO `.txt` label import.
- Automatic approval copy into approved image/label storage.
- Dataset version creation with train/val/test splits and `data.yaml`.
- Training run queue records and cancellation records.
- Celery/Redis training worker that executes Ultralytics YOLO jobs one at a time.
- Model registry listing, artifact viewing, and single active model activation.
- Detection runs for mixed image/video uploads using the active model.
- Detection media preview, prediction records, video timestamps, and output media storage.
- Static frontend served by FastAPI.

Architectural or partially modeled, but not fully automated yet:

- Evaluation worker and metric import.
- Model deployment gate.
- Production-grade job monitor and retry/cancel controls across all workers.

## Architecture Overview

The current application is a metadata API plus local artifact storage:

```text
Browser / Static Frontend
  -> FastAPI application
  -> SQLAlchemy session layer
  -> PostgreSQL metadata tables
  -> Local storage folders
  -> OpenCV/imagehash frame extraction
  -> CVAT API for annotation tasks
  -> Dataset builder for YOLO-compatible datasets
  -> Redis/Celery training and detection queues
  -> Ultralytics YOLO training and detection workers
```

Core runtime modules:

- `app/main.py`: creates the FastAPI app, includes routers, and mounts the frontend.
- `app/api/`: REST endpoints grouped by workflow area.
- `app/core/config.py`: environment-backed settings.
- `app/core/database.py`: SQLAlchemy engine/session setup.
- `app/models/`: SQLAlchemy table mappings.
- `app/schemas/`: Pydantic request/response models.
- `app/services/frame_extraction.py`: video sampling, blur scoring, perceptual hashing, and frame writes.
- `app/services/cvat_client.py`: CVAT API client.
- `app/workers/celery_app.py`: Celery app configured from `REDIS_URL`.
- `app/workers/training_worker.py`: YOLO training task runner.
- `app/workers/detection_worker.py`: YOLO detection task runner for images and videos.
- `scripts/schema.sql`: PostgreSQL schema.
- `scripts/README.md`: database table architecture and relationships.
- `frontend/`: static HTML/CSS/JS shell served from `/`.

## Repository Layout

```text
logo_detection_app/
+-- app/
|   +-- main.py
|   +-- api/
|   |   +-- annotation.py
|   |   +-- classes.py
|   |   +-- datasets.py
|   |   +-- frames.py
|   |   +-- health.py
|   |   +-- projects.py
|   |   +-- training.py
|   |   +-- uploads.py
|   |   +-- models.py
|   |   +-- detection.py
|   +-- core/
|   |   +-- config.py
|   |   +-- database.py
|   +-- models/
|   |   +-- dataset.py
|   |   +-- image.py
|   |   +-- model_registry.py
|   |   +-- project.py
|   |   +-- training_run.py
|   |   +-- detection.py
|   +-- schemas/
|   +-- services/
|   +-- workers/
|       +-- celery_app.py
|       +-- training_worker.py
|       +-- detection_worker.py
+-- frontend/
|   +-- index.html
|   +-- app.js
|   +-- styles.css
|   +-- README.md
+-- scripts/
|   +-- schema.sql
|   +-- README.md
+-- storage/
|   +-- raw_uploads/
|   +-- extracted_frames/
|   +-- imported_labels/
|   +-- approved_images/
|   +-- datasets/
|   +-- training_runs/
|   +-- models/
|   +-- detection_runs/
+-- README.md
```

## Training Worker

Run Redis, the FastAPI app, and the Celery worker in separate terminals.

```bash
redis-server
```

```bash
cd /home/gitto/social_media/logo_detection_app
../venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
cd /home/gitto/social_media/logo_detection_app
../venv/bin/celery -A app.workers.celery_app.celery_app worker --loglevel=info --concurrency=1
```

Use `--concurrency=1` for one YOLO training run at a time. This avoids GPU/CPU memory contention. Additional workers can be added later per GPU.

The root repository currently keeps Python dependencies in `/home/gitto/social_media/requirements.txt`.

## Prerequisites

- Python 3.12 or compatible Python 3 version.
- PostgreSQL reachable through `DATABASE_URL`.
- Python dependencies from the repository root `requirements.txt`.
- CVAT, if using the CVAT task endpoints.
- OpenCV-compatible video files, if using frame extraction.

## Step-by-Step Local Setup

Run commands from the repository root unless noted.

1. Create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment variables.

Create `logo_detection_app/.env` with values like:

```env
DATABASE_URL=postgresql+psycopg2://logo_user:logo_password@localhost:5432/logo_db
STORAGE_ROOT=/home/gitto/social_media/logo_detection_app/storage
ACTIVE_MODEL_PATH=/home/gitto/social_media/logo_detection_app/storage/models/latest/best.pt
REDIS_URL=redis://localhost:6379/0
CVAT_URL=http://localhost:8080
CVAT_USERNAME=admin
CVAT_PASSWORD=admin
CVAT_ACCESS_TOKEN=
CVAT_VERIFY_SSL=true
CVAT_ORGANIZATION=
YOLO_BASE_MODEL=yolo11m.pt
YOLO_IMAGE_SIZE=1280
YOLO_EPOCHS=100
YOLO_BATCH=8
YOLO_PATIENCE=25
```

4. Create the database schema.

```bash
psql "$DATABASE_URL" -f logo_detection_app/scripts/schema.sql
```

If your shell cannot use the SQLAlchemy-style URL, use a standard PostgreSQL URL or connect with `psql` flags and run the same SQL file.

5. Start the FastAPI server.

```bash
cd logo_detection_app
uvicorn app.main:app --reload
```

6. Open the app.

```text
http://127.0.0.1:8000/
```

7. Check health.

```bash
curl http://127.0.0.1:8000/health
```

## Step-by-Step Product Workflow

### 1. Create a Project

```bash
curl -X POST http://127.0.0.1:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"project_name":"BMW Campaign","description":"Initial upload batch"}'
```

The response includes the project `id`. Use that id in later steps.

### 2. Configure Classes

Classes are stable YOLO IDs used by CVAT tasks, imported labels, dataset `data.yaml`, training, and detection output.

```bash
curl http://127.0.0.1:8000/classes
curl -X POST http://127.0.0.1:8000/classes \
  -H "Content-Type: application/json" \
  -d '{"class_name":"logo","is_active":true}'
```

The API assigns the next available `class_id`. Keep existing IDs stable after training starts.

### 3. Upload Images or Videos

```bash
curl -X POST http://127.0.0.1:8000/projects/1/upload \
  -F "files=@/path/to/image.jpg" \
  -F "files=@/path/to/video.mp4"
```

Uploaded files are copied to:

```text
storage/raw_uploads/{project_id}/
```

Database rows are created in `logo_images` with:

- `source_type = image` for images.
- `source_type = video` for videos.
- `status = uploaded`.

### 4. Extract Unique Frames from Videos

```bash
curl -X POST http://127.0.0.1:8000/projects/1/extract-frames \
  -H "Content-Type: application/json" \
  -d '{"frame_interval_seconds":1.0,"hash_threshold":5,"blur_threshold":80.0}'
```

The frame extraction service:

1. Opens each uploaded video with OpenCV.
2. Samples a candidate frame every `frame_interval_seconds`.
3. Computes a blur score using Laplacian variance.
4. Rejects frames below `blur_threshold`.
5. Computes a perceptual hash.
6. Rejects frames whose hash distance is within `hash_threshold` of an already saved frame.
7. Writes accepted frames as JPG files.
8. Creates `logo_images` rows with `source_type = frame` and `status = ready_for_annotation`.

Extracted frames are written to:

```text
storage/extracted_frames/{project_id}/
```

### 5. Review Annotation Candidates

```bash
curl http://127.0.0.1:8000/projects/1/annotation-items
curl http://127.0.0.1:8000/images/{image_id}/preview
```

Annotation candidates include uploaded images and extracted frames that have annotation-related statuses.

### 6. Send Items to CVAT

Check CVAT connectivity:

```bash
curl http://127.0.0.1:8000/cvat/status
```

Send all ready items:

```bash
curl -X POST http://127.0.0.1:8000/projects/1/send-to-cvat \
  -H "Content-Type: application/json" \
  -d '{}'
```

Send selected images only:

```bash
curl -X POST http://127.0.0.1:8000/projects/1/send-to-cvat \
  -H "Content-Type: application/json" \
  -d '{"image_ids":[10,11,12]}'
```

The app creates a CVAT task, uploads the image files, records the task in `logo_cvat_tasks`, and marks the items as `sent_to_cvat`.

Optional task actions:

```bash
curl -X POST http://127.0.0.1:8000/projects/1/cvat-tasks/{task_row_id}/auto-annotate
curl -X POST http://127.0.0.1:8000/projects/1/cvat-tasks/{task_row_id}/sync-labels
```

### 7. Export YOLO Labels from CVAT

After annotation in CVAT, export labels in YOLO format. Each label file must have the same stem as the corresponding image or frame.

Example:

```text
item-0116_frame_000390_30bf5910.jpg
item-0116_frame_000390_30bf5910.txt
```

### 8. Import YOLO Labels

```bash
curl -X POST http://127.0.0.1:8000/projects/1/import-yolo-labels \
  -F "files=@/path/to/item-0116_frame_000390_30bf5910.txt"
```

The app validates each label line, writes labels to:

```text
storage/imported_labels/{project_id}/
```

Then it copies valid image/label pairs to:

```text
storage/approved_images/{project_id}/images/
storage/approved_images/{project_id}/labels/
```

The associated image row is marked `approved`.

### 9. Approve or Reject Manually

Approve an image that already has a label:

```bash
curl -X PATCH http://127.0.0.1:8000/images/{image_id}/approve
```

Reject an item:

```bash
curl -X PATCH http://127.0.0.1:8000/images/{image_id}/reject
```

### 10. Build a Dataset Version

```bash
curl -X POST http://127.0.0.1:8000/datasets/build \
  -H "Content-Type: application/json" \
  -d '{"dataset_name":"dataset_v1","project_id":1,"train_pct":80,"val_pct":10,"test_pct":10,"description":"First approved BMW dataset"}'
```

The builder uses approved image/label pairs and writes:

```text
storage/datasets/{dataset_name}/
+-- images/
|   +-- train/
|   +-- val/
|   +-- test/
+-- labels/
|   +-- train/
|   +-- val/
|   +-- test/
+-- data.yaml
```

### 11. Create a Training Run

```bash
curl -X POST http://127.0.0.1:8000/training-runs \
  -H "Content-Type: application/json" \
  -d '{"dataset_version_id":1,"run_name":"logo_v1","base_model_path":"yolo11m.pt","image_size":1280,"epochs":100,"batch_size":8,"patience":25,"device":"auto"}'
```

This creates a queued training run record and prepares:

```text
storage/training_runs/{run_name}/
+-- train.log
+-- weights/
    +-- best.pt
```

The Celery training worker executes the YOLO job, updates status and timestamps, writes metrics, and registers a candidate model when training succeeds.

### 12. Activate a Model

```bash
curl http://127.0.0.1:8000/models
curl -X PATCH http://127.0.0.1:8000/models/{model_id}/activate
```

Only one model can be active at a time. Activating a model marks it `deployed`, archives the previous deployed model, and records `deployed_at`.

### 13. Run Detection

```bash
curl -X POST http://127.0.0.1:8000/detection-runs \
  -F "run_name=mall_entry_test" \
  -F "description=Smoke test" \
  -F "confidence=0.25" \
  -F "iou=0.70" \
  -F "image_size=1280" \
  -F "device=auto" \
  -F "files=@/path/to/image.jpg" \
  -F "files=@/path/to/video.mp4"
```

Detection uses the active model, stores the run under `storage/detection_runs/{run_name}/`, queues the Celery detection worker, saves output media, and records per-box prediction rows in `logo_predictions`.

## API Reference

### Health

```text
GET /health
GET /dashboard/summary
```

### Classes

```text
GET   /classes
GET   /classes/summary
POST  /classes
PATCH /classes/{class_row_id}
```

### Projects

```text
POST   /projects
GET    /projects
GET    /projects/{project_id}
PATCH  /projects/{project_id}
DELETE /projects/{project_id}
```

### Uploads

```text
POST /projects/{project_id}/upload
GET  /projects/{project_id}/files
```

Supported upload extensions:

```text
.mp4 .mov .avi .mkv .webm .jpg .jpeg .png
```

### Frames

```text
POST /projects/{project_id}/extract-frames
GET  /projects/{project_id}/frames
GET  /projects/{project_id}/frames/page
```

Frame extraction request:

```json
{
  "frame_interval_seconds": 1.0,
  "hash_threshold": 5,
  "blur_threshold": 80.0
}
```

### Annotation and CVAT

```text
GET   /projects/{project_id}/annotation-items
GET   /projects/{project_id}/annotation-items/page
GET   /approved-images
GET   /cvat/status
POST  /projects/{project_id}/send-to-cvat
GET   /projects/{project_id}/cvat-tasks
POST  /projects/{project_id}/cvat-tasks/{task_row_id}/auto-annotate
POST  /projects/{project_id}/cvat-tasks/{task_row_id}/sync-labels
POST  /projects/{project_id}/import-yolo-labels
PATCH /images/{image_id}/approve
PATCH /images/{image_id}/reject
GET   /images/{image_id}/preview
GET   /images/{image_id}/labels
```

### Datasets

```text
GET  /datasets
POST /datasets/build
```

Dataset build request:

```json
{
  "dataset_name": "dataset_v1",
  "project_id": 1,
  "train_pct": 80,
  "val_pct": 10,
  "test_pct": 10,
  "description": "First dataset version"
}
```

### Training Runs

```text
GET   /training-runs
POST  /training-runs
PATCH /training-runs/{run_id}/cancel
```

Training run request:

```json
{
  "dataset_version_id": 1,
  "run_name": "logo_v1",
  "base_model_path": "yolo11m.pt",
  "image_size": 1280,
  "epochs": 100,
  "batch_size": 8,
  "patience": 25,
  "device": "auto"
}
```

### Models

```text
GET   /models
PATCH /models/{model_id}/activate
GET   /models/{model_id}/artifacts/{artifact_name}
```

Supported artifact names:

```text
confusion_matrix
pr_curve
f1_curve
results
```

### Detection

```text
GET   /detection-runs
POST  /detection-runs
GET   /detection-runs/{run_id}
PATCH /detection-runs/{run_id}/start
GET   /detection-inputs/{input_id}/media
```

## Storage Layout

The storage root defaults to:

```text
/home/gitto/social_media/logo_detection_app/storage
```

It can be overridden with `STORAGE_ROOT`.

Important directories:

```text
storage/raw_uploads/{project_id}/
```

Original uploaded images and videos.

```text
storage/extracted_frames/{project_id}/
```

Unique, non-blurry frames extracted from uploaded videos.

```text
storage/imported_labels/{project_id}/
```

YOLO label files imported from CVAT or another annotation tool.

```text
storage/approved_images/{project_id}/images/
storage/approved_images/{project_id}/labels/
```

Approved image/label pairs eligible for dataset building and training.

```text
storage/datasets/{dataset_name}/
```

YOLO-compatible dataset versions.

```text
storage/training_runs/{run_name}/
```

Training logs, YOLO outputs, weights, metrics files, and graph artifacts.

```text
storage/detection_runs/{run_name}/
```

Detection inputs, detected output images/videos, and media served by the detection results UI.

```text
storage/models/latest/best.pt
```

Optional active production model path setting.

## File Formats

### Uploaded Images

Supported image extensions:

```text
.jpg .jpeg .png
```

Image rows store:

- `image_path`: absolute path to the image file.
- `source_type`: `image`.
- `status`: initially `uploaded`.

### Uploaded Videos

Supported video extensions:

```text
.mp4 .mov .avi .mkv .webm
```

Video rows store:

- `image_path`: absolute path to the uploaded video file.
- `source_type`: `video`.
- `source_video_path`: same as `image_path`.
- `status`: initially `uploaded`.

### Extracted Frames

Extracted frame files are JPG images. Current naming pattern:

```text
{video_stem}_frame_{frame_number:06d}.jpg
```

Frame rows store:

- `image_path`: absolute path to the extracted JPG.
- `source_type`: `frame`.
- `source_video_path`: absolute path to the source video.
- `frame_number`: original frame index in the source video.
- `blur_score`: Laplacian variance score.
- `status`: `ready_for_annotation`.

### YOLO Label Files

YOLO labels are plain text `.txt` files. The label file stem must match the image/frame stem.

Each non-empty line must contain exactly five values:

```text
class_id x_center y_center width height
```

Rules enforced by the API:

- `class_id` must be an integer.
- `class_id` must be non-negative.
- `x_center`, `y_center`, `width`, and `height` must be floats.
- Coordinates must be normalized between `0` and `1`.

Example:

```text
0 0.512 0.438 0.144 0.087
1 0.224 0.733 0.091 0.052
```

### YOLO Dataset Format

Dataset versions follow the Ultralytics/YOLO layout:

```text
dataset_v1/
+-- images/
|   +-- train/
|   +-- val/
|   +-- test/
+-- labels/
|   +-- train/
|   +-- val/
|   +-- test/
+-- data.yaml
```

Generated `data.yaml` format:

```yaml
path: /home/gitto/social_media/logo_detection_app/storage/datasets/dataset_v1
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: logo
```

Class names come from active rows in `logo_classes`, ordered by `class_id`. If no active classes exist, the app uses a default single class named `logo`.

## Database Architecture

Tables are defined in `scripts/schema.sql`, mapped in `app/models/`, and extended by the detection API for detection-specific tables/columns when needed.

For the full table-by-table architecture, relationships, status values, and storage mapping, see:

```text
scripts/README.md
```

### `logo_projects`

Stores project-level metadata.

Important columns:

- `id`
- `project_name`
- `description`
- `status`
- `created_at`
- `updated_at`

Relationships:

- One project has many `logo_images`.

### `logo_classes`

Stores the class registry used in CVAT tasks and dataset `data.yaml`.

Important columns:

- `class_id`: stable numeric YOLO class id.
- `class_name`: class label.
- `is_active`: whether the class is included in new annotation tasks and datasets.

Architecture rule: keep class IDs stable across dataset versions. Do not reorder class IDs between training runs.

### `logo_images`

Stores uploaded videos, uploaded images, extracted frames, label paths, review status, and dataset split assignment.

Important columns:

- `project_id`
- `image_path`
- `label_path`
- `source_type`: `image`, `video`, or `frame`.
- `source_video_path`
- `frame_number`
- `status`
- `is_duplicate`
- `duplicate_of`
- `blur_score`
- `split`: `train`, `val`, or `test` once used in a dataset.

### `logo_cvat_tasks`

Stores CVAT task references created by the app.

Important columns:

- `project_id`
- `cvat_task_id`
- `task_name`
- `task_url`
- `status`
- `sent_count`

### `logo_dataset_versions`

Stores immutable dataset version metadata.

Important columns:

- `version_name`
- `data_yaml_path`
- `train_count`
- `val_count`
- `test_count`
- `class_distribution`
- `status`

### `logo_dataset_images`

Join table between dataset versions and approved image rows.

Important columns:

- `dataset_version_id`
- `image_id`
- `split`

Constraint:

- `uq_logo_dataset_image` prevents the same image from being added twice to the same dataset version.

### `logo_training_runs`

Stores training job metadata.

Important columns:

- `run_name`
- `dataset_version_id`
- `base_model_path`
- `output_model_path`
- `status`
- `training_config`
- `error_message`
- `started_at`
- `finished_at`

### `logo_training_metrics`

Stores model evaluation metrics and deployment decisions.

Important columns:

- `map50`
- `map50_95`
- `precision_score`
- `recall_score`
- `per_class_metrics`
- `confusion_matrix_path`
- `pr_curve_path`
- `decision`
- `decision_reason`

### `logo_model_registry`

Stores candidate, approved, deployed, and archived model records.

Important columns:

- `model_name`
- `model_path`
- `training_run_id`
- `dataset_version_id`
- `status`
- `is_active`
- `deployed_at`

Constraint:

- `uq_logo_model_registry_one_active` allows only one active model.

### `logo_predictions`

Stores detection and inference outputs.

Important columns:

- `image_path`
- `detection_run_id`
- `detection_input_id`
- `model_id`
- `training_run_id`
- `class_id`
- `class_name`
- `confidence`
- `bbox`
- `frame_number`
- `timestamp_seconds`
- `decision`

### `logo_detection_runs`

Stores one detection batch created from the active model.

Important columns:

- `run_name`
- `description`
- `model_id`
- `training_run_id`
- `status`
- `config`
- `input_count`
- `image_count`
- `video_count`
- `output_dir`
- `error_message`
- `started_at`
- `finished_at`

### `logo_detection_inputs`

Stores each uploaded image/video inside a detection run.

Important columns:

- `detection_run_id`
- `file_name`
- `file_type`
- `input_path`
- `output_path`
- `status`
- `frame_count`
- `duration_seconds`

## Configuration

Settings are loaded by `app/core/config.py` using Pydantic settings. The `.env` file is read from the `logo_detection_app/` directory.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://logo_user:logo_password@localhost:5432/logo_db` | SQLAlchemy database connection |
| `STORAGE_ROOT` | `/home/gitto/social_media/logo_detection_app/storage` | Artifact storage root |
| `ACTIVE_MODEL_PATH` | `/home/gitto/social_media/logo_detection_app/storage/models/latest/best.pt` | Intended production model path |
| `REDIS_URL` | `redis://localhost:6379/0` | Future background job broker |
| `CVAT_URL` | `http://localhost:8080` | CVAT server |
| `CVAT_USERNAME` | `admin` | CVAT username |
| `CVAT_PASSWORD` | `admin` | CVAT password |
| `CVAT_ACCESS_TOKEN` | unset | Optional CVAT token |
| `CVAT_VERIFY_SSL` | `true` | CVAT TLS verification |
| `CVAT_ORGANIZATION` | unset | Optional CVAT organization slug |
| `YOLO_BASE_MODEL` | `yolo11m.pt` | Default base model for training |
| `YOLO_IMAGE_SIZE` | `1280` | Default training image size |
| `YOLO_EPOCHS` | `100` | Default training epochs |
| `YOLO_BATCH` | `8` | Default training batch size |
| `YOLO_PATIENCE` | `25` | Default early stopping patience |

## Status Model

Project statuses:

```text
created
```

Image and frame statuses used by the app:

```text
uploaded
ready_for_annotation
sent_to_cvat
annotation_done
approved
rejected
```

Additional statuses referenced by the architecture:

```text
frame_extracted
duplicate
rejected_blurry
```

Dataset statuses:

```text
created
```

Training run statuses:

```text
queued
running
completed
failed
cancelled
```

Model registry statuses:

```text
candidate
evaluated
approved
rejected
deployed
archived
```

## Frontend

The frontend is static and mounted by FastAPI from `frontend/`:

```text
http://127.0.0.1:8000/
```

The app is mounted after API routers, so API paths continue to resolve normally and all other root paths are served as static frontend files.

## Target Production Architecture

The intended end-to-end workflow is:

```text
Create project
  -> Upload images/videos
  -> Extract unique frames
  -> Send images/frames to CVAT
  -> Import approved YOLO labels
  -> Build dataset version from all approved data
  -> Train YOLO model from base weights
  -> Evaluate on validation/test split
  -> Compare against active production model
  -> Approve, reject, or deploy candidate
  -> Serve active model through inference API
  -> Store prediction history
```

Recommended production components:

| Component | Role |
|---|---|
| FastAPI | API and frontend static serving |
| PostgreSQL | Metadata, statuses, metrics, registry |
| Local storage or S3/MinIO | Images, videos, labels, datasets, models, reports |
| OpenCV + imagehash | Frame extraction and duplicate filtering |
| CVAT | Human annotation workflow |
| Ultralytics YOLO | Object detection training and validation |
| Celery + Redis | Long-running frame extraction, dataset, training, detection, and evaluation jobs |
| Model registry table | Candidate/deployed model tracking |
| Detection API | Batch image/video logo detection |
| Job monitor | Central visibility for queued, running, completed, and failed jobs |

Recommended future endpoints:

```text
POST /training-runs/{run_id}/metrics
PATCH /models/{model_id}/approve
GET  /jobs
GET  /jobs/{job_id}
PATCH /jobs/{job_id}/retry
PATCH /jobs/{job_id}/cancel
```

Recommended YOLO training command for a worker:

```bash
yolo detect train \
  model=yolo11m.pt \
  data=storage/datasets/dataset_v1/data.yaml \
  imgsz=1280 \
  epochs=100 \
  batch=8 \
  patience=25 \
  project=storage/models/runs \
  name=logo_v1
```

Recommended evaluation command:

```bash
yolo detect val \
  model=storage/models/runs/logo_v1/weights/best.pt \
  data=storage/datasets/dataset_v1/data.yaml \
  imgsz=1280 \
  split=test \
  project=storage/reports \
  name=logo_v1_eval
```

Deployment gate examples:

- New overall `mAP50` should not fall below the active model by more than `0.01`.
- New precision should not fall below the active model by more than `0.02`.
- New recall should not fall below the active model by more than `0.03`.
- Important classes should not suffer large per-class recall or mAP drops.
- False positives should be reviewed before deployment.

## Operational Notes

- Keep `logo_classes.class_id` stable forever once training starts.
- Keep label stems matched to image stems for YOLO label import.
- Dataset names and training run names are sanitized to alphanumeric, dash, and underscore characters.
- Dataset versions should be treated as immutable once created.
- Every training cycle should normally use all approved data, not only newly uploaded data.
- Avoid putting duplicate frames across train/val/test splits.
- Prefer a stable test set once the project matures.
- The current dataset split is deterministic by approved image id order, not randomized.
- The current frame extraction endpoint runs synchronously inside the API request.
- Large video extraction, training, and evaluation should move to background workers before production use.
