import torch.nn.functional as F  # cross entropy loss 사용

def sequence_cross_entropy(logits, targets):
    # logits shape: (B, T, vocab_size)
    # cross_entropy는 (B, C, T) 형태를 기대하므로 차원을 바꿔줌
    return F.cross_entropy(logits.transpose(1, 2), targets)

def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()  # 학습 모드로 전환
    total_loss, total_count = 0.0, 0  # epoch 평균 loss 계산용 변수
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)  # 입력과 정답을 CPU/GPU로 이동
        logits = model(xb)  # forward
        loss = sequence_cross_entropy(logits, yb)  # loss 계산
        optimizer.zero_grad()  # gradient 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 파라미터 업데이트

        # 평균 loss 계산을 위해 누적
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        # step 수 제한이 있으면 해당 step까지만 학습
        if max_steps is not None and step + 1 >= max_steps:
            break

    return total_loss / total_count  # epoch 평균 loss 반환

def train(model, loader, optimizer, device, num_epochs=100, max_steps_per_epoch=None):
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=max_steps_per_epoch)  # 한 epoch 학습
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")  # 학습 loss 출력
    return model  # 학습 완료된 모델 반환