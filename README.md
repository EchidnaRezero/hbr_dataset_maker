# HBR Character BBox Auto-Labeling Pipeline

> 이미지 생성 모델 학습(LoRA 파인튜닝 · MOSS-Image 등 Layout-Conditioned 데이터 포맷)을 위해  
> 헤븐 번즈 레드(55명) 캐릭터를 인스턴스(BBox) 단위로 검출 · 분류 · 필터링하는 파이프라인입니다.

[![Showcase](https://img.shields.io/badge/Showcase-2997ff?style=for-the-badge&logo=githubpages&logoColor=white)](https://echidnarezero.github.io/hbr_dataset_maker/)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-GPU-005CED?logo=onnx&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-WAL-003B57?logo=sqlite&logoColor=white)

---

## Project Structure

```text
.
├── index.html               # GitHub Pages 정적 쇼케이스 (단일 파일)
├── src/hbr_pipeline/        # 핵심 라이브러리 (detector, classifier, phash, db)
├── scripts/
│   ├── collect/             # Danbooru 공식 API 메타·이미지 수집 CLI
│   ├── pipeline/            # DB 빌드 · 신규 이미지 적재 · Inference · v3 학습 · HF 배포
│   └── eval/                # OOD 정량 평가 · Embedding · Grad-CAM 분석
├── models/                  # v3 Linear Probe weights (~630 KB)
├── docs/
│   ├── problems/            # 문제별 기술 문서 (01 ~ 03)
│   ├── analysis/            # 1,536-D Embedding 공간 분석 문서
│   └── assets/              # 쇼케이스·문서용 시각화 이미지
└── data/ & outputs/         # 로컬 DB · 원본 이미지 · 산출물 (Git 제외, HF Gated 백업)
```

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# 1) 임의 폴더 Inference + BBox 시각화
python scripts/pipeline/predict_folder.py --input-dir <input_dir> --out-dir outputs/eval_v3

# 2) 신규 이미지를 5-Tier Safety-Valve + pHash 중복제거 거쳐 Master DB 적재
python scripts/pipeline/ingest_images.py --input-dir <input_dir> --source official

# 3) Master DB 통계 조회
python scripts/pipeline/db_cli.py --stats --top-chars 20
```

<details>
<summary><b>세부 CLI (v3 재학습 · Grad-CAM · Hugging Face 복원)</b></summary>

```bash
# v3 Linear Head 재학습 (CPU ~22초)
python scripts/pipeline/train_v3.py \
  --v2-embs data/embeddings/v2_solo_embeddings.npz \
  --multi-embs data/embeddings/danbooru_multi_crops_for_v3.npz \
  --out-weights models/v3_pure_hbr55_plus_multi_camie.npz

# 정량 평가 및 Grad-CAM 분석
python scripts/eval/eval_hard_testset.py --weights models/v3_pure_hbr55_plus_multi_camie.npz
python scripts/eval/gradcam_v3_failures.py
python scripts/eval/compare_gradcam_v1_v3.py
```

**Hugging Face Gated Repo 복원** (`Kamome33/hbr-character-classifier`, `Kamome33/hbr-dataset-master`):
```python
import os
from huggingface_hub import snapshot_download

token = os.environ["HF_TOKEN"]  # 또는 `huggingface-cli login`
snapshot_download("Kamome33/hbr-character-classifier", repo_type="model", local_dir="models", token=token)
snapshot_download("Kamome33/hbr-dataset-master", repo_type="dataset", local_dir="data/db", token=token)
```

</details>
