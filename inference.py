

import torch  # PyTorch import
import torch.nn.functional as F  # softmax를 사용하기 위한 함수형 API import



@torch.no_grad()  # 추론 시 gradient 계산 비활성화
def sample_gpt(model, info, device, start_text="ROMEO:", max_new_tokens=400):
    block_size = info["block_size"]  # 모델이 한 번에 볼 수 있는 최대 문맥 길이
    stoi = info["stoi"]  # 문자 -> 인덱스 사전
    itos = info["itos"]  # 인덱스 -> 문자 사전

    model.eval()  # dropout 등을 비활성화하는 평가 모드로 전환

    # 초기 context를 0으로 채운다. shape: (배치 크기 1, block_size)
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    
    # 시작 문자열을 context 뒤쪽에 순서대로 넣는다
    for ch in start_text:
        if ch in stoi:  # vocab에 있는 문자만 처리
            ix = torch.tensor([[stoi[ch]]], device=device)  # 문자 하나를 인덱스 텐서로 변환
            context = torch.cat([context[:, 1:], ix], dim=1)  # 왼쪽 한 칸 밀고 새 문자 추가

    out = list(start_text)  # 생성 결과를 문자 리스트로 저장

    # max_new_tokens 개수만큼 새 문자 생성
    for _ in range(max_new_tokens):
        logits = model(context)  # 현재 문맥에 대한 로짓 계산
        logits = logits[:, -1, :]  # 마지막 위치의 예측만 사용
        temperature = 0.8  
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)  # 로짓을 확률로 변환
        ix = torch.multinomial(probs, num_samples=1)  # 확률에 따라 다음 문자 샘플링
        out.append(itos[ix.item()])  # 인덱스를 문자로 바꿔 결과에 추가
        context = torch.cat([context[:, 1:], ix], dim=1)  # 생성한 문자를 context에 반영

    return "".join(out)  # 문자 리스트를 하나의 문자열로 합쳐 반환