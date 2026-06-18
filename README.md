# Tiny GPT

이 프로젝트는 문자 단위(character-level) Tiny GPT를 학습부터 추론까지 나누어 구현한 예제입니다. 전체 흐름은 data/input.txt를 읽고, 문자를 숫자 토큰으로 바꾼 뒤, Transformer 기반 모델을 학습하고, 학습된 모델로 새 문장을 생성하는 방식입니다.

## 현재 폴더 구조

현재 코드는 `semi_gpt/tiny-GPT` 폴더 바로 아래에 있습니다.

```text
tiny-GPT/
├─ data/
│  └─ input.txt
├─ dataset.py
├─ inference.py
├─ main.py
├─ model.py
├─ train.py
├─ util.py
└─ README.md
```

## 파일 역할

- `data/input.txt`: 학습에 사용하는 텍스트 데이터입니다. 현재는 Tiny Shakespeare 데이터입니다.
- `dataset.py`: 다음 문자 예측용 데이터셋인 `NextTokenDataset`을 정의합니다.
- `util.py`: 텍스트 로드, 문자 사전 생성, 인코딩, `DataLoader` 생성을 담당합니다.
- `model.py`: TinyGPT 모델과 Attention, FeedForward, Transformer Block을 정의합니다.
- `train.py`: loss 계산과 학습 루프를 담당합니다.
- `inference.py`: 학습된 모델로 텍스트를 생성하는 `sample_gpt()`를 담고 있습니다.
- `main.py`: 데이터 로드부터 학습, 샘플 생성까지 전체 흐름을 실행합니다.

## 전체 흐름

1. `main.py`가 `load_dataloader("./data/input.txt", ...)`를 호출합니다.
2. `util.py`가 텍스트를 읽고 고유 문자 목록을 만듭니다.
3. 문자와 숫자를 서로 바꾸는 사전을 만듭니다.
   - `stoi`: 문자에서 숫자로 변환합니다.
   - `itos`: 숫자에서 문자로 변환합니다.
4. 전체 텍스트를 숫자 토큰으로 인코딩합니다.
5. `NextTokenDataset`이 입력 `x`와 정답 `y`를 만듭니다.
   - `x`: 현재 문맥입니다.
   - `y`: 한 글자 뒤로 밀린 정답 문장입니다.
6. `TinyGPT`가 각 위치에서 다음 문자의 확률을 예측합니다.
7. `train.py`의 학습 루프가 cross entropy loss를 줄이도록 모델을 업데이트합니다.
8. `inference.py`의 `sample_gpt()`가 한 글자씩 샘플링하면서 새 텍스트를 생성합니다.

## 실행 방법

`tiny-GPT` 폴더에서 실행합니다.

```powershell
python main.py
```

현재 `main.py`는 아래 경로를 기준으로 데이터를 읽습니다.

```python
load_dataloader("./data/input.txt", block_size=64, batch_size=64)
```

따라서 다른 폴더에서 실행하면 `./data/input.txt`를 찾지 못할 수 있습니다. 가장 안전한 방법은 `tiny-GPT` 폴더로 이동한 뒤 실행하는 것입니다.

```powershell
cd "C:\Users\HW Hye\Python\7-1\semi_gpt\tiny-GPT"
python main.py
```

## 예상 출력

처음에는 모델 출력 크기를 확인합니다.

```text
logits.shape: torch.Size([64, 64, vocab_size])
```

그 다음 epoch마다 학습 loss가 출력되고, 마지막에 `ROMEO:`로 시작하는 샘플 문장이 생성됩니다.

초기 결과는 문법이 어색할 수 있습니다. 이 모델은 작은 문자 단위 모델이고, 학습 시간과 모델 크기에 따라 생성 품질이 크게 달라집니다.

## 주요 하이퍼파라미터

현재 코드 기준 주요 설정은 다음과 같습니다.

- `block_size=64`
- `batch_size=64`
- `emb_dim=128`
- `num_heads=4`
- `num_layers=4`
- `dropout=0.1`
- `lr=3e-4`
- `num_epochs=100`
- `max_steps_per_epoch=300`

실험할 때는 `block_size`, `emb_dim`, `num_layers`, `num_heads`, `num_epochs`를 바꿔보면 모델 성능과 학습 속도의 차이를 보기 좋습니다.

## 학습 목적

이 프로젝트는 완성도 높은 GPT 구현이라기보다 GPT 내부 구조를 직접 이해하기 위한 학습용 코드에 가깝습니다. 특히 아래 개념을 확인하기 좋습니다.

- 문자 토큰화
- 토큰 임베딩
- 포지션 임베딩
- causal self-attention
- multi-head attention
- residual connection
- layer normalization
- feed-forward network
- autoregressive sampling

## 정리

이 프로젝트는 문자 단위 Tiny GPT를 직접 구현하면서, 데이터 전처리부터 학습과 추론까지의 전체 흐름을 이해하기 위한 실습용 예제입니다. 복잡한 대규모 언어모델보다는 Transformer 기반 언어모델의 핵심 구조를 학습하는 데 초점을 두었습니다.
