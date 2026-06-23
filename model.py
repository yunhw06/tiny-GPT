import torch  # PyTorch import
import torch.nn as nn  # 신경망 모듈 import
import torch.nn.functional as F  # softmax 등 함수형 API import

class Head(nn.Module):  # 단일 self-attention head
    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)  # 입력을 key 벡터로 변환
        self.query = nn.Linear(emb_dim, head_size, bias=False)  # 입력을 query 벡터로 변환
        self.value = nn.Linear(emb_dim, head_size, bias=False)  # 입력을 value 벡터로 변환

        # 미래 토큰을 보지 못하게 하는 causal mask를 미리 저장
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)  # attention weight에 적용할 dropout

    def forward(self, x):
        B, T, C = x.shape  # B: batch, T: sequence length, C: embedding dimension
        k = self.key(x)  # key 계산
        q = self.query(x)  # query 계산
        v = self.value(x)  # value 계산

        
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5) # attention score 계산
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # 현재 시점 이후의 미래 토큰은 보지 못하도록 mask 적용
        wei = F.softmax(wei, dim=-1) # attention score를 확률분포로 변환
        wei = self.dropout(wei) # dropout 적용
        out = wei @ v # attention weight와 value를 곱해 출력 생성
        return out

class MultiHeadAttention(nn.Module):  # 여러 attention head를 병렬로 사용하는 모듈
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = emb_dim // num_heads  # 각 head가 담당할 차원 수
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])  
        self.proj = nn.Linear(emb_dim, emb_dim)  
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  
        out = self.proj(out)  
        out = self.dropout(out) 
        return out

class FeedForward(nn.Module):  
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),  
            nn.ReLU(),  
            nn.Linear(4 * emb_dim, emb_dim),  
            nn.Dropout(dropout),  
        )

    def forward(self, x):
        return self.net(x)  

class Block(nn.Module):  
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)  
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)  
        self.ln2 = nn.LayerNorm(emb_dim)  
        self.ffwd = FeedForward(emb_dim, dropout)  

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  
        x = x + self.ffwd(self.ln2(x))  
        return x 

class TinyGPT(nn.Module):  # 전체 GPT 스타일 언어모델
    def __init__(self, info, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        vocab_size = info["vocab_size"]  # 문자 집합 크기
        block_size = info["block_size"]  # 최대 문맥 길이
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)  # 문자 토큰을 임베딩 벡터로 바꾸는 층
        self.position_embedding = nn.Embedding(block_size, emb_dim)  # 위치 정보를 임베딩 벡터로 바꾸는 층
        
        # 여러 개의 Transformer block을 순차적으로 쌓음
        self.blocks = nn.Sequential(*[  
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(emb_dim)  # 마지막 layer norm
        self.lm_head = nn.Linear(emb_dim, vocab_size)  # 최종 hidden state를 vocab 크기의 로짓으로 바꾸는 출력층

    def forward(self, x):
        B, T = x.shape  # 입력 shape: (batch_size, sequence_length)
        pos = torch.arange(T, device=x.device)  # 현재 시퀀스 길이에 맞는 위치 인덱스 생성
        tok = self.token_embedding(x)  # 토큰 임베딩 계산
        pos = self.position_embedding(pos)[None]  # 위치 임베딩 계산 후 batch 차원에 맞게 확장
        h = tok + pos  # 토큰 정보와 위치 정보를 더해 입력 표현 생성
        h = self.blocks(h)  # 여러 Transformer block 통과
        h = self.ln_f(h)  # 마지막 layer norm 적용
        logits = self.lm_head(h)  # 각 위치별 다음 문자 예측 점수 계산
        return logits
