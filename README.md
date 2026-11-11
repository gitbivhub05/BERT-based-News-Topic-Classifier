# BERT News Article Topic Classification

Full-stack starter: React + TypeScript + Vite frontend, FastAPI backend, and Hugging Face BERT training scripts.

> This is application/training code, not a trained project-specific model or a copy of your dataset. Train the model and export artifacts before using `/predict`.

## Structure
- `backend/app`: FastAPI API, request schemas, BERT inference service
- `backend/scripts`: training and evaluation scripts
- `frontend`: React dashboard and article classifier
- `data`: put your CSV here (not committed)
- `backend/artifacts`: generated model files (not committed)

## Dataset
CSV columns must be `text` and `label`. Labels should be contiguous integer IDs starting at 0. This project does not guess topic names for IDs 0–3; verify the source label mapping.

## Backend (Python 3.10/3.11)
From the project root:
```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
API docs: http://127.0.0.1:8000/docs

The API can start without a model. In that case `/health` reports `model_loaded: false`, and prediction returns HTTP 503 until artifacts are available.

## Train BERT
From project root:
```bash
pip install -r backend/requirements.txt
python backend/scripts/train_bert.py --csv data/news_train.csv --output-dir backend/artifacts/bert-news-classifier
```
Run `python backend/scripts/train_bert.py --help` for options. Default split is stratified 80/10/10; validation macro-F1 selects the checkpoint, and the held-out test split is evaluated after training. Review exact/near duplicates and source leakage before treating results as final. GPU strongly recommended. The same script can be run in Colab with the correct CSV path.

## Frontend
In another terminal:
```bash
cd frontend
npm install
# copy .env.example to .env if you need to change the API URL
npm run dev
```
Open http://localhost:5173. Default API URL: `http://127.0.0.1:8000`.

## API
- `GET /health`
- `GET /model-info`
- `POST /predict` with `{"text":"Article text"}`
- `POST /predict-batch` with `{"texts":["Article one","Article two"]}`

Prediction returns numeric class ID, model probabilities, and inference time. No confidence values are fabricated.

## Research cautions
- Do not invent class names; verify the label map.
- Keep test data untouched during model selection.
- Report accuracy, macro-F1, per-class precision/recall, split sizes, seed, model, tokenizer and hyperparameters.
- A 20-article external sample is supplementary, not a replacement for the internal test set.
- A 1%-sample pilot result is not a full-dataset final result.

## Push to GitHub
Create an empty repository, then:
```bash
git init
git add .
git commit -m "Initial BERT news topic classification project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Do not commit datasets, model weights, `.env`, virtual environments or secrets.

## Deployment
Deploy frontend as a static site and FastAPI as a Python service. Set frontend `VITE_API_BASE_URL` to the deployed API URL; set backend `CORS_ORIGINS` to the frontend origin. Confirm hosting memory/disk can accommodate BERT.
