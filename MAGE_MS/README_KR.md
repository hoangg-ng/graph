# MAGE + TimeMixer: 서버 병렬 평가 실행 꾸러미

**대상:** `hoangg-ng/graph`의 `Jiwon-work` 브랜치, `MAGE_MS/` 폴더. 기존 GPU 평가 꾸러미는 추가 파일 기반입니다. 이 문서에 추가된 후속 모형은 `proposal_model.py`, `train_eval.py`, `train_proposal.py`에 별도 arm으로 연결하지만, 원본 MAGE 구현과 `run_matched_eval.py`는 수정하지 않습니다.

기준: `MAGE Proposal.pdf`의 필수 실험 및 `proposal.md`의 96→12 예측 조건. 비용 측정은 현재 환경의 NVIDIA RTX 3090에서 수행했으며, 추가 모형 학습은 실행하지 않았습니다. 다른 GPU에서는 지연 시간·메모리가 달라질 수 있습니다.

## 0. 새 파일 위치

다운로드한 압축파일 안의 `.py`, `.sh` 파일을 모두 서버의 아래 폴더에 넣습니다.

```
/home/user/Workspace/KIMJIWON/graph/MAGE_MS/
```

- `train_eval.py`: 기존 모형을 이용해 학습, 검증·시험, 재개용 저장, 평가 수치 기록. engineered features는 **각 96시점 입력 구간 내부**에서 생성합니다.
- `run_eval.py`: GPU 0–3 동시 사용, 비어 있는 GPU 메모리 확인, 실행·중단·재시작, 각 작업 로그.
- `run_eval.sh`: 서버 Conda Python으로 실행하는 편의용 실행 파일.
- `tune_shared.py`: Shared만 이용한 **검증 자료 기반** 10회 학습 설정 탐색(시험 자료를 사용하지 않음).
- `report_eval.py`: MAE·RMSE·MAPE·시점별 오차·메모리·소요 시간 집계 및 Proposed–Shared/Reversed의 사전 정의된 판정.
- `power_diag.py`: 학습된 양의 공간 연산자에 대해 \(k=1,2,4,8\) 거듭제곱의 차이를 측정.
- `test_eval_tools.py`: 평가 도구 및 누출 방지 단위 시험.
- `profile_models.py`, `report_complexity.py`: 기존 체크포인트의 추론 비용을 측정하고 기존 MAE와 결합.
- `run_followup_eval.py`: 추가 모형 40개 학습을 기본 dry-run으로 계획하고, `--execute`일 때만 실행.
- `test_complexity.py`: 비용 계측과 추가 모형의 CPU 단위 시험.

## 1. 서버 확인

VS Code Remote-SSH 터미널에서 실행합니다.

```bash
cd /home/user/Workspace/KIMJIWON/graph/MAGE_MS
export PATH="/home/user/yes/envs/jiwon-mage/bin:$PATH"
python -c "import torch; print(torch.__version__); print(torch.cuda.device_count())"
nvidia-smi
```

`torch.cuda.device_count()`가 4로 나오는지 확인합니다. 연구실 공용 GPU에 기존 작업이 있다면 스케줄러가 자유 메모리와 GPU 사용률을 확인한 뒤 비어 있는 GPU에만 새로운 작업을 배치합니다. 기존 작업을 강제로 종료하지 않습니다.

원본 MAGE 모듈이 안 보이면 저장소 루트에서 `git submodule update --init --recursive`를 실행해야 할 수 있습니다. 현재 `MAGE_MS/official_repro/` 경로도 사용 가능합니다.

## 2. 사전 검증

```bash
cd /home/user/Workspace/KIMJIWON/graph/MAGE_MS
python -m unittest test_proposal test_proposal_prep test_eval_tools test_complexity
python run_eval.py --profile full --dry-run
```

마지막 명령은 **실험 43개를 표시만** 하고 GPU를 사용하지 않습니다.

데이터는 다음 파일을 필요로 합니다.

```
MAGE_MS/data_proposal96/PEMS04/his.npz
MAGE_MS/data_proposal96/PEMS04/idx_train.npy
MAGE_MS/data_proposal96/PEMS04/idx_val.npy
MAGE_MS/data_proposal96/PEMS04/idx_test.npy
```

`PEMS08/`에도 같은 네 개의 파일이 필요합니다. GitHub에는 `metadata.json`만 있는 경우가 있으므로 실제 서버에 `his.npz`가 있는지 확인해야 합니다.

준비되지 않았다면 원본 자료가 저장된 폴더를 확인하고 예를 들어 다음처럼 생성합니다.

```bash
python prep_proposal.py --datasets 04 08 \
  --raw_root /home/user/Workspace/KIMJIWON/graph/MAGE_MS/raw/ASTGNN/data \
  --out data_proposal96
```

입력 원본은 `${raw_root}/PEMS04/PEMS04.npz`와 `${raw_root}/PEMS08/PEMS08.npz`에 있어야 합니다. 실제 원본 자료 경로가 다르면 `--raw_root`를 바꾸십시오. 기준 프로그램은 시간 순서 60/20/20 분할, 학습 구간 정규화, 양수 정답 마스킹을 사용합니다.

## 3. GPU 자원 시험: 1에포크 실행

```bash
python run_eval.py \
  --profile smoke \
  --gpus 0 1 2 3 \
  --slots-per-gpu 1 \
  --threads-per-job 3 \
  --epochs 1 --patience 1 \
  --results-dir results/eval_smoke \
  --logs-dir logs/eval_smoke \
  --preflight-tests
```

GPU 4개에 각각 최대 1개의 학습 작업을 실행합니다. GPU 메모리 최소 여유는 기본 12 GiB, GPU 1개당 예상 학습 메모리는 9 GiB, 여분은 3 GiB로 설정했습니다. **1에포크 결과는 성능 지표로 사용하지 않고 실행 가능 여부만 확인합니다.**

## 4. Shared 모형만으로 학습 설정 결정

제안서는 핵심 모형들을 비교하기 전에 Shared 모형의 검증 성능으로 학습률·가중치 감쇠·드롭아웃을 선택하고, 모든 최종 실험에 같은 값을 사용하도록 규정합니다.

```bash
python tune_shared.py --gpus 0 1 2 3 \
  --epochs 60 --patience 15 --threads-per-job 3
```

10개의 설정을 PEMS04 Shared의 seed 0으로 시험합니다. 선택에는 **검증 MAE만 사용하며 시험 자료에 대한 예측도 계산하지 않습니다.** 다음 파일을 생성합니다.

```
results/tune_shared/selected_recipe.json
```

각 학습 설정의 결과와 원본 코드·자료의 지문도 저장합니다. 이후 코드를 수정하거나 전처리 자료를 바꾸면 해당 탐색 결과는 다시 만들어야 합니다.

## 5. 초기 성능·메모리 측정: 3회

```bash
python run_eval.py \
  --profile pilot \
  --gpus 0 1 2 3 \
  --slots-per-gpu 1 \
  --threads-per-job 3 \
  --recipe results/tune_shared/selected_recipe.json \
  --results-dir results/eval_proposal96 \
  --logs-dir logs/eval_proposal96
```

PEMS04의 Shared, Proposed, Reversed 각 seed 0을 실행합니다. 작업별 GPU 최대 메모리와 에포크당 소요 시간을 확인하여 24 GiB GPU에서 적절한 동시 실행 수를 결정합니다.

```bash
python report_eval.py --results-dir results/eval_proposal96 --profile pilot
```

`results/eval_proposal96/summary.md`에서 3개 초기 결과를 확인할 수 있습니다. 제안서의 통계 판정은 이 단계에서는 하지 않습니다.

## 6. 본 평가: 43회

```bash
python run_eval.py \
  --profile full \
  --gpus 0 1 2 3 \
  --slots-per-gpu 1 \
  --threads-per-job 3 \
  --recipe results/tune_shared/selected_recipe.json \
  --epochs 300 --patience 30 \
  --results-dir results/eval_proposal96 \
  --logs-dir logs/eval_proposal96
```

- PEMS04: Shared·Proposed·Reversed·Independent·Uniform × 5 seeds = **25회**.
- PEMS04: 동일한 96→12 조건의 MAGE 및 MAGE + engineered features × 3 seeds = **6회**.
- PEMS08: Shared·Proposed·Reversed·Uniform × 3 seeds = **12회**.
- **총 43회**, pilot에서 끝난 작업은 자동 건너뛰며 기존 실험의 중단 저장 파일이 있으면 이어서 학습합니다.

모든 정규 실험에는 동일한 학습률, 가중치 감쇠, 드롭아웃, 배치 크기, 조기 종료 조건을 사용합니다. 필요에 따라 `--bs`, `--tf32`, `--slots-per-gpu` 값을 바꿀 수 있지만 **이러한 조건은 최종 실험 중 변경하면 안 됩니다.** 변경 시 기존 결과 폴더와 분리하여 다시 학습해야 합니다.

안전하게 더 높은 병렬성을 시험하려면 pilot의 GPU 최대 메모리를 먼저 확인하십시오. 충분히 작고 GPU 사용률이 포화되지 않은 경우에만 `--slots-per-gpu 2`를 검토할 수 있습니다. GPU 메모리 부족이 발생하면 먼저 배치 크기를 낮추고 **모든 비교 모형에 같은 배치 크기를 적용**해야 합니다.

`run_eval.py`는 학습 시작 전에 `nvidia-smi`로 GPU 사용률과 자유 메모리를 확인합니다. 이용 가능한 GPU가 없는 상태가 30분간 지속되면 실행을 중지하고, 아직 시작하지 않은 작업은 다음 실행에서 처리하도록 남깁니다. 이미 완료한 결과는 유지됩니다.

### 화면을 닫아도 작업을 유지하려는 경우

연구실 서버에 `tmux`가 설치되어 있다면 터미널에서 아래처럼 실행할 수 있습니다.

```bash
tmux new -s mage-eval
# tmux 안에서 위의 본 평가 명령 실행
# Ctrl+B, 이어서 D 로 분리
```

다시 접속할 때는 `tmux attach -t mage-eval`을 사용합니다. 이 작업은 **서버에서 사용자 본인이 실행하는 과정**입니다.

## 7. 실행 상태·로그 확인

```bash
python run_eval.py --profile full --status-only
nvidia-smi
# 작업별 학습 로그
tail -n 30 logs/eval_proposal96/PEMS04_proposed_s0.log
```

- `results/eval_proposal96/PEMS04_proposed_s0.json`: 평가 지표와 실험 설정·자료 지문.
- `results/eval_proposal96/PEMS04_proposed_s0.pt`: 검증 MAE가 가장 낮았던 모형.
- `<result>.json.resume.pt`: 진행 중 학습의 매 10에포크 저장 상태. 완료 후 삭제.
- `logs/eval_proposal96/events.jsonl`: 작업 시작·종료·실패 기록.

서버가 재시작되거나 작업이 중단되더라도 같은 명령을 실행하면 원본 코드·자료·학습 설정이 일치하는 결과를 자동 건너뛰고, 중단된 작업은 가장 최근에 저장한 에포크부터 다시 학습합니다.

**결과가 원본 코드나 자료 지문과 불일치하면 자동으로 합치지 않고 오류를 출력합니다.** 이때는 새 결과 폴더를 사용하십시오.

## 8. 최종 집계

```bash
python report_eval.py \
  --results-dir results/eval_proposal96 \
  --profile full
```

다음 파일이 생성됩니다.

- `summary.md`: 핵심 모형들의 평균±표준편차, 시점별 오차, 자원 사용량, 판정, 미완료 작업.
- `summary.csv`: 원시 집계 표.
- `paired_comparisons.csv`: Proposed 대 Shared/Reversed의 seed별 MAE 차이.
- `decision.json`: **양쪽 비교 모두** 5 seeds 중 4회 이상 승리하고 평균 MAE가 1% 이상 개선되었는지 판정.
- `missing_jobs.csv`: 남은 실험 목록.

`report_eval.py`는 코드 버전이 섞이거나 데이터 분할이 다른 결과를 발견하면 `INVALID_COMPARISON`으로 표시합니다. 다섯 시드가 모두 확보되지 않으면 `INCOMPLETE`로 표시합니다. 비교 기준이 충족되지 않으면 `NOT_SUPPORTED_BY_PRESPECIFIED_RULE`로 표시합니다. 이는 반드시 효과가 0이라는 의미는 아닙니다.

## 9. 학습된 공간 연산자의 거듭제곱 차이 확인

```bash
python power_diag.py \
  --dataset PEMS04 --arm proposed --seed 0 \
  --results-dir results/eval_proposal96 \
  --device cuda:0 --samples 12
```

학습된 각 층의 \(S_+\)에 대해 상대 Frobenius 거리 \(D(1,2)\), \(D(2,4)\), \(D(4,8)\), \(D(1,8)\)을 측정합니다. 제안서에서는 여러 고정된 시험 구간에 대한 평균을 요구하므로 기본값을 12개 구간으로 설정했습니다. 가능하면 seed 0–4 모두 같은 명령을 실행하여 시드 간 차이도 확인하십시오. 이 검사는 진단 목적이므로 \(N\times N\) 행렬을 일시적으로 생성합니다(PEMS04 노드 307개). **학습 자체는 기존 저계수 연산자를 사용합니다.**

## 10. 아직 포함되지 않은 실험

이 꾸러미는 제안서의 **핵심 5개 비교 모형과 96→12 MAGE 두 기준 모형에 대한 평가를 자동화**합니다. 다음은 별도 구현 또는 기존 연구 코드 연동이 필요합니다.

- TimeMixer **독립 기준 모형**의 공식 구현으로 동일 자료 분할에서 학습·평가.
- STID **독립 기준 모형**의 공식 구현으로 동일 자료 분할에서 학습·평가.
- MAGE의 원래 **12→12** 설정 재현(이전 `official_repro` 결과와 분리 보고).
- 선택 실험: learned schedule, synthetic, 추가 자료, block bootstrap.

위 세 가지 공식 기준 실험까지 완료했다고 보고해서는 안 됩니다. 파일 형식과 모델별 검증 설정이 같다는 사실을 확인한 후 같은 평가 표에 추가해야 합니다.

## 11. 계산량 비교와 추가 비교 모형

기존 체크포인트를 복원하여 재학습 없이 FP32 추론 비용을 측정합니다. 측정 입력은 검증 구간에서 가져오며 시험 정답은 사용하지 않습니다. 기존 결과는 그대로 두고 새 파일은 `results/complexity_v1/` 아래에 저장합니다.

```bash
python profile_models.py \
  --datasets PEMS04 PEMS08 \
  --arms mage shared proposed reversed independent uniform \
  --seeds 0 \
  --results-dir results/eval_proposal96 \
  --data-dir data_proposal96 \
  --out-dir results/complexity_v1 \
  --device cuda:0 \
  --batch-sizes 1 64 \
  --warmup 20 --repeats 100

python report_complexity.py \
  --mae-results-dir results/eval_proposal96 \
  --profile-dir results/complexity_v1
```

보고서는 매개변수, 등록된 연산 기준 FLOPs, 그래프 적용 횟수, batch 1/64 지연 시간·GPU 메모리, 처리량을 기존 5시드 MAE와 결합합니다. FLOP 계측기는 등록된 연산만 셀 수 있으므로 결과에는 계산 방법과 커버리지 주의사항이 함께 기록됩니다. 그래프 적용을 동일한 기준으로 세기 어려운 원본 MAGE는 `not comparable`로 표시합니다. 지연 시간과 메모리는 GPU·PyTorch 버전 및 동시 GPU 작업에 영향을 받습니다.

`power_diag.py`의 \(N\times N\) 행렬 진단 결과도 `results/complexity_v1/diagnostics/`에 따로 저장되며, 추론 FLOPs에 포함되지 않습니다.

추가 모형은 `mage_features`, `no_spatial`, `grouped`, `proposed_no_pdm`입니다. `mage_features` 특징은 각 과거 창 안에서만 생성됩니다. 새 학습은 기본적으로 실행하지 않으며, 먼저 40개 작업 목록을 확인하십시오.

```bash
python run_followup_eval.py --dry-run
```

실제 추가 학습이 승인된 후에만 `--execute`를 붙입니다. 선택 recipe의 `lr=0.005`, `wd=0.0001`, `dropout=0.3`을 사용하며, recipe 출처·원본 코드 지문과 현재 worker 코드 지문을 결과에 기록합니다. 현재 코드 지문이 recipe 생성 시점과 다르면 그 차이를 숨기지 않습니다. 새 결과와 로그는 `results/eval_followup96/`, `logs/eval_followup96/`에 저장됩니다.

## 연구실 서버에서 주의할 점

1. 공용 GPU의 기존 사용자 작업을 강제 중단하지 않습니다. 충분한 자유 메모리가 없으면 스케줄러는 대기하거나 중지합니다.
2. 논문에 보고할 시험 자료 성능은 **학습 설정을 고른 후** 한 번의 고정된 절차로 계산합니다. 중간에 시험 MAE를 보고 설정을 조정하지 않습니다.
3. `MAGE_MS/results/step8`의 이전 결과는 예전 모형 실험이므로 이 꾸러미의 신규 결과와 자동으로 혼합되지 않습니다.
4. `profile_models.py`는 선택한 GPU에서 동기화된 이벤트 기반 추론 시간을 측정합니다. 다른 사용자의 GPU 작업이 있을 때는 측정값 변동 가능성을 기록하고 반복 측정을 고려하십시오.
