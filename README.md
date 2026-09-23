# SpotCSE

SpotCSE는 한 문장씩 담긴 텍스트 파일로 문장 임베딩 모델을 학습하고 SentEval로 평가합니다. 이 저장소의 예제는 단일 GPU 또는 CPU에서 실행됩니다.
준비물: Git, curl, wget, tar, 인터넷 연결, uv 의존성과 모델을 저장할 디스크 공간.

## 1. 저장소 받기

저장소를 클론한 뒤 프로젝트 폴더로 이동합니다.

```bash
git clone https://github.com/bluejun0901/SVD-text-embedding.git SVD-text-embedding
cd SVD-text-embedding
```

이미 파일을 받았다면 프로젝트 루트(`pyproject.toml`이 있는 폴더)로 이동하면 됩니다. 아래 명령은 모두 그 위치를 기준으로 합니다.

## 2. uv와 Python 환경 설치

`uv`가 없다면 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 설치합니다. Linux/macOS에서는 다음 명령을 사용할 수 있습니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

새 터미널을 열어 `uv --version`으로 설치를 확인한 뒤, 잠금 파일에 지정된 패키지를 설치합니다. `uv`는 `.python-version`에 지정된 Python 3.9를 사용합니다.

```bash
uv sync --locked
source .venv/bin/activate
```

첫 설치에서는 PyTorch와 Python 다운로드 때문에 시간이 걸릴 수 있습니다.

## 3. 데이터 받기

학습용 Wikipedia 문장과 평가용 SentEval 데이터를 각각 받습니다. 다운로드 스크립트는 실행한 위치에 파일을 저장하므로 아래처럼 해당 디렉터리에서 실행하세요.

```bash
(cd data && bash download_wiki.sh)
(cd SentEval/data/downstream && bash download_dataset.sh)
```

`data/wiki1m_for_simcse.txt`는 **한 줄에 문장 하나**인 학습 파일입니다. 자체 데이터를 사용할 때도 이 형식을 지키고 `SPOTCSE_TRAIN_FILE`로 경로를 지정할 수 있습니다. SentEval 데이터는 주기적 검증과 학습 종료 후 평가에 사용됩니다.

## 4. 예제 실행

```bash
./run_spotcse_example.sh
```

스크립트는 기본적으로 `bert-base-uncased` 체크포인트, GPU 0(사용 가능할 때), Wikipedia 학습 파일을 사용합니다. GPU가 없으면 CPU로 실행됩니다. 최초 실행에서는 모델 가중치를 다운로드합니다. 전체 데이터 1회 학습과 주기적 평가는 오래 걸릴 수 있습니다.

결과 모델과 로그는 `result/spotcse/`에 저장됩니다. 출력 디렉터리가 이미 차 있다면 새 경로를 지정하세요.

```bash
SPOTCSE_OUTPUT_DIR=result/spotcse_run2 ./run_spotcse_example.sh
```

다른 GPU, 모델, 학습 파일은 환경변수로 선택합니다.

```bash
SPOTCSE_GPU=1 SPOTCSE_MODEL=/path/to/bert-or-roberta-checkpoint \
SPOTCSE_TRAIN_FILE=data/my_sentences.txt ./run_spotcse_example.sh
```

추가 학습 인자는 스크립트 뒤에 붙일 수 있습니다. 예를 들어 GPU에서 혼합 정밀도를 사용하려면 `./run_spotcse_example.sh --fp16`을 실행합니다. 지원 인자는 `uv run --locked python train.py --help`에서 볼 수 있습니다.

이 서버의 시스템 CUDA 라이브러리 경로가 PyTorch와 충돌하는 경우를 피하기 위해, 예제 스크립트는 Python 실행 시 `LD_LIBRARY_PATH`를 제외합니다. 분산 학습, TPU, DeepSpeed, Apex는 지원하지 않습니다.
