# TinyGPT Character-Level Language Model

문자 단위(Character-level) GPT 스타일 언어모델을 PyTorch로 구현한 프로젝트입니다.  
텍스트 파일을 입력으로 받아 문자 단위 vocabulary를 만들고, Transformer Decoder 구조를 이용해 다음 문자를 예측하도록 학습합니다.

---

## 1. 프로젝트 개요

이 프로젝트는 간단한 GPT 구조를 직접 구현하여 다음 과정을 이해하는 것을 목표로 합니다.

- 텍스트 데이터를 문자 단위로 토큰화
- 입력 시퀀스와 정답 시퀀스 생성
- Token Embedding과 Position Embedding 구성
- Causal Self-Attention 구현
- Multi-Head Attention 구현
- Transformer Block 구성
- Cross Entropy Loss 기반 다음 문자 예측 학습
- Autoregressive 방식의 텍스트 생성

즉, 전체 구조는 다음과 같습니다.

```text
Text File
→ Character Vocabulary
→ Integer Token Sequence
→ NextTokenDataset
→ TinyGPT Model
→ Cross Entropy Training
→ Autoregressive Text Generation
```

---

## 2. 파일 구조

```text
.
├── dataset.py       # next-token prediction용 Dataset 정의
├── util.py          # 텍스트 로드, 문자 vocabulary 생성, DataLoader 생성
├── model.py         # TinyGPT 모델 및 Transformer 구성 요소 정의
├── train.py         # 학습 루프 및 loss 함수 정의
├── inference.py     # 학습된 모델로 텍스트 생성
├── main.py          # 전체 실행 코드
└── data/
    └── novel.txt    # 학습할 텍스트 파일
```

---

## 3. 주요 파일 설명

### 3.1 `dataset.py`

`NextTokenDataset`은 전체 텍스트를 정수 인덱스 시퀀스로 받은 뒤, 길이 `block_size`의 입력 `x`와 이를 한 칸 오른쪽으로 민 정답 `y`를 반환합니다.

```python
x = self.data[idx : idx + self.block_size]
y = self.data[idx + 1 : idx + self.block_size + 1]
```

예를 들어 데이터가 다음과 같고,

```python
data = [10, 20, 30, 40, 50]
block_size = 3
```

하나의 샘플은 다음처럼 만들어집니다.

```python
x = [10, 20, 30]
y = [20, 30, 40]
```

이는 각 위치에서 다음 문자를 예측하도록 학습하기 위한 구조입니다.

---

### 3.2 `util.py`

`load_dataloader()` 함수는 텍스트 파일을 읽고, 문자 단위 vocabulary를 생성합니다.

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
```

여기서 중요한 점은 이 모델이 **단어 단위가 아니라 문자 단위 모델**이라는 것입니다.

예를 들어 텍스트가 다음과 같다면,

```text
별 헤는 밤
```

모델은 이를 단어가 아니라 문자 단위로 처리합니다.

```text
별 / 공백 / 헤 / 는 / 공백 / 밤
```

`stoi`는 문자를 정수로 바꾸는 사전이고, `itos`는 정수를 다시 문자로 바꾸는 사전입니다.

---

### 3.3 `model.py`

`model.py`에는 TinyGPT 모델과 Transformer 구성 요소가 들어 있습니다.

주요 구성 요소는 다음과 같습니다.

| 클래스 | 역할 |
|---|---|
| `Head` | 단일 causal self-attention head |
| `MultiHeadAttention` | 여러 attention head를 병렬로 사용 |
| `FeedForward` | 각 위치별 비선형 변환 |
| `Block` | Transformer decoder block |
| `TinyGPT` | 전체 GPT 스타일 언어모델 |

전체 모델 흐름은 다음과 같습니다.

```text
Input Token IDs
→ Token Embedding
+ Position Embedding
→ Transformer Blocks
→ Final LayerNorm
→ Linear LM Head
→ Logits
```

---

## 4. 모델 구조

### 4.1 Token Embedding

```python
self.token_embedding = nn.Embedding(vocab_size, emb_dim)
```

문자 인덱스를 `emb_dim` 차원의 벡터로 변환합니다.

입력 shape이 다음과 같다면,

```python
x.shape = (B, T)
```

Token Embedding 이후 shape은 다음과 같습니다.

```python
tok.shape = (B, T, emb_dim)
```

---

### 4.2 Position Embedding

```python
self.position_embedding = nn.Embedding(block_size, emb_dim)
```

Self-Attention은 순서 정보를 직접 알 수 없기 때문에, 각 위치에 대한 position embedding을 더해줍니다.

```python
h = tok + pos
```

이를 통해 모델은 같은 문자라도 위치에 따라 다르게 이해할 수 있습니다.

---

### 4.3 Causal Self-Attention

`Head` 클래스에서는 Query, Key, Value를 이용해 scaled dot-product attention을 계산합니다.

```python
wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
```

수식으로는 다음과 같습니다.

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k))V
```

---

### 4.4 Causal Mask

GPT는 현재 위치에서 미래 토큰을 보면 안 됩니다.  
따라서 lower triangular mask를 사용합니다.

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

예를 들어 sequence length가 4라면 mask는 다음과 같습니다.

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

즉 각 위치는 자기 자신과 과거 위치만 볼 수 있습니다.

---

### 4.5 Multi-Head Attention

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)
```

여러 attention head를 병렬로 사용하면, 각 head가 서로 다른 문맥 관계를 학습할 수 있습니다.

예를 들어 `emb_dim=128`, `num_heads=4`라면 각 head의 크기는 다음과 같습니다.

```python
head_size = 128 // 4 = 32
```

각 head의 출력은 `(B, T, 32)`이고, 4개를 concat하면 `(B, T, 128)`이 됩니다.

---

### 4.6 FeedForward Network

```python
nn.Linear(emb_dim, 4 * emb_dim)
nn.ReLU()
nn.Linear(4 * emb_dim, emb_dim)
```

Attention이 토큰 간 정보를 섞는 역할을 한다면, FeedForward Network는 각 위치의 hidden representation을 비선형적으로 변환하는 역할을 합니다.

---

### 4.7 Transformer Block

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

이 구조는 다음 두 가지 특징을 가집니다.

1. **Residual Connection**  
   원래 입력을 보존하면서 attention과 FFN의 출력만 더합니다.

2. **Pre-LayerNorm**  
   Attention과 FFN 이전에 LayerNorm을 적용합니다.

---

## 5. 학습 과정

### 5.1 Loss 함수

`train.py`에서는 Cross Entropy Loss를 사용합니다.

```python
return F.cross_entropy(logits.transpose(1, 2), targets)
```

모델 출력 `logits`의 shape은 다음과 같습니다.

```python
logits.shape = (B, T, vocab_size)
```

하지만 PyTorch의 `cross_entropy`는 class dimension이 두 번째 차원에 와야 하므로 다음처럼 변환합니다.

```python
logits.transpose(1, 2)
```

변환 후 shape은 다음과 같습니다.

```python
(B, vocab_size, T)
```

정답 `targets`의 shape은 다음과 같습니다.

```python
targets.shape = (B, T)
```

즉 각 위치마다 실제 다음 문자 index를 정답으로 사용합니다.

---

### 5.2 학습 루프

```python
logits = model(xb)
loss = sequence_cross_entropy(logits, yb)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

학습 과정은 다음 순서로 진행됩니다.

1. 입력 batch를 모델에 넣어 logits 계산
2. logits와 target을 비교해 loss 계산
3. 기존 gradient 초기화
4. 역전파로 gradient 계산
5. optimizer로 parameter 업데이트

---

## 6. 추론 과정

`inference.py`의 `sample_gpt()` 함수는 학습된 모델을 사용해 문자를 하나씩 생성합니다.

```python
logits = model(context)
logits = logits[:, -1, :]
probs = F.softmax(logits, dim=-1)
ix = torch.multinomial(probs, num_samples=1)
```

중요한 점은 모델이 전체 context에 대해 logits를 출력하지만, 생성 시에는 **마지막 위치의 logits만 사용한다는 것**입니다.

```python
logits = logits[:, -1, :]
```

그 이유는 현재 context 뒤에 올 다음 문자 하나만 필요하기 때문입니다.

---

### 6.1 Temperature

```python
temperature = 0.8
logits = logits / temperature
```

Temperature는 생성 확률분포의 sharpness를 조절합니다.

| Temperature | 생성 경향 |
|---|---|
| 낮음 | 더 결정적이고 반복적인 출력 |
| 1.0 | 기본 softmax 분포 |
| 높음 | 더 다양하지만 불안정한 출력 |

---

### 6.2 Multinomial Sampling

```python
ix = torch.multinomial(probs, num_samples=1)
```

가장 확률이 높은 문자를 무조건 선택하는 것이 아니라, 확률분포에 따라 랜덤하게 다음 문자를 선택합니다.

---

### 6.3 Context 업데이트

```python
context = torch.cat([context[:, 1:], ix], dim=1)
```

모델은 최대 `block_size` 길이의 문맥만 볼 수 있기 때문에, 새 문자를 생성할 때마다 가장 오래된 문자를 제거하고 새 문자를 뒤에 붙입니다.

예를 들어,

```text
[A, B, C, D]
```

에서 새 문자 `E`가 생성되면 다음 context는 다음과 같습니다.

```text
[B, C, D, E]
```

---

## 7. 실행 방법

### 7.1 데이터 준비

학습할 텍스트 파일을 다음 경로에 저장합니다.

```text
./data/novel.txt
```

예시:

```text
data/
└── novel.txt
```

---

### 7.2 실행

```bash
python main.py
```

실행하면 다음 과정이 진행됩니다.

1. 텍스트 파일 로드
2. 문자 vocabulary 생성
3. DataLoader 생성
4. TinyGPT 모델 생성
5. 모델 forward shape 확인
6. 모델 학습
7. 텍스트 생성
8. 생성 결과를 `logt.txt`에 저장

---

## 8. 주요 하이퍼파라미터

`main.py`에서 다음 값을 조절할 수 있습니다.

```python
block_size = 64
batch_size = 64
num_epochs = 30
max_steps_per_epoch = 300
learning_rate = 3e-4
```

`model.py`에서는 다음 값을 조절할 수 있습니다.

```python
emb_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1
```

| 하이퍼파라미터 | 의미 |
|---|---|
| `block_size` | 모델이 한 번에 볼 수 있는 최대 문맥 길이 |
| `batch_size` | 한 번에 학습하는 샘플 수 |
| `num_epochs` | 전체 학습 반복 횟수 |
| `learning_rate` | optimizer 학습률 |
| `emb_dim` | embedding dimension |
| `num_heads` | attention head 개수 |
| `num_layers` | Transformer block 개수 |
| `dropout` | dropout 비율 |

---

## 9. 주의사항

### 9.1 이 모델은 문자 단위 모델입니다

현재 코드는 단어 단위 tokenizer를 사용하지 않습니다.

```python
chars = sorted(list(set(text)))
```

따라서 모델은 단어가 아니라 문자 단위로 학습합니다.

---

### 9.2 데이터가 작으면 쉽게 암기할 수 있습니다

짧은 시 한 편이나 짧은 텍스트만 학습시키면 모델은 일반적인 언어 능력을 배우기보다 원문을 암기할 가능성이 큽니다.

학습이 부족하면 이상한 글자 조합이 나오고, 학습이 과하면 원문을 거의 그대로 생성할 수 있습니다.

---

### 9.3 Validation set이 없습니다

현재 코드는 train loss만 계산합니다.  
따라서 모델이 실제로 일반화하는지, 단순히 학습 데이터를 외우는지 확인하기 어렵습니다.

개선하려면 train/validation split을 추가하고 validation loss를 함께 확인하는 것이 좋습니다.

---

### 9.4 초기 context가 0으로 채워집니다

```python
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
```

현재 구현에서는 별도의 padding token이 없기 때문에, `0`은 실제 문자 index입니다.  
따라서 초기 context가 특정 문자 반복으로 해석될 수 있습니다.

---

### 9.5 학습 데이터에 없는 문자는 추론에서 무시됩니다

```python
if ch in stoi:
```

`start_text`에 학습 데이터에 없는 문자가 포함되어 있으면 해당 문자는 context에 들어가지 않습니다.

---

## 10. 예시 출력

학습이 완료되면 `sample_gpt()` 함수를 통해 다음과 같이 텍스트를 생성할 수 있습니다.

```python
generated_text = sample_gpt(
    model,
    info,
    device,
    start_text="설렁탕을:",
    max_new_tokens=500
)
```

생성된 결과는 콘솔에 출력되고, `logt.txt` 파일에도 저장됩니다.

---

## 11. 라이선스 및 목적

이 코드는 교육용 TinyGPT 구현입니다.  
실제 대규모 GPT 모델과 달리 문자 단위 tokenizer와 작은 Transformer 구조를 사용하며, Transformer 언어모델의 핵심 원리를 이해하기 위한 목적으로 작성되었습니다.
