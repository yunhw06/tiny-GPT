import torch  # PyTorch import
from util import load_dataloader  # 데이터 로더 생성 함수 import
from inference import sample_gpt  # 텍스트 생성 함수 import
from model import TinyGPT  # TinyGPT 모델 클래스 import
from train import train  # 학습 함수 import

if __name__ == "__main__":  # 현재 파일을 직접 실행할 때만 아래 코드를 수행
    
    loader, info = load_dataloader("./data/poet.txt", block_size=64, batch_size=64)
    # data/poet.txt 파일을 읽어서 DataLoader와 모델에 필요한 정보(info)를 생성
    # block_size=64는 한 번에 보는 문맥 길이
    # batch_size=64는 한 번에 학습하는 샘플 수

    xb, yb = next(iter(loader))
    # DataLoader에서 첫 번째 배치를 하나 가져옴
    # xb는 입력 시퀀스, yb는 정답 시퀀스

    model = TinyGPT(info)
    # TinyGPT 모델 객체 생성
    # info 안에는 vocab_size, block_size 같은 모델 생성에 필요한 정보가 들어 있음
    
    logits = model(xb)
    # 첫 번째 배치를 모델에 넣어서 forward 연산 수행
    # logits는 각 위치에서 다음 문자에 대한 점수(score) 출력

    print("logits.shape:", logits.shape)
    # 모델 출력의 shape 확인
    # 보통 (batch_size, block_size, vocab_size) 형태가 나옴

    device = "cuda" if torch.cuda.is_available() else "cpu"  
    # GPU를 사용할 수 있으면 cuda, 아니면 cpu를 사용

    model = TinyGPT(info).to(device)
    # 모델을 다시 새로 생성한 뒤 선택한 device로 이동
    # 주의: 위에서 만든 model과는 다른 새 모델임

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # AdamW 옵티마이저 생성
    # 학습률은 3e-4로 설정

    model = train(model, loader, optimizer, device, num_epochs=100, max_steps_per_epoch=300)
    # 모델 학습 수행
    # num_epochs=100: 전체 데이터 반복 횟수
    # max_steps_per_epoch=300: 한 epoch당 최대 300 step까지만 학습

    generated_text = sample_gpt(model, info, device, start_text="별:", max_new_tokens=500)
    print(generated_text)
    with open("logp.txt", "w", encoding="utf-8") as log_file:
        log_file.write(generated_text + "\n")
    # 학습된 모델을 이용해 "경제:"으로 시작하는 텍스트를 생성
    # 최대 500개의 새 토큰(문자)을 생성해서 출력
    # 생성된 텍스트를 콘솔 화면에 출력하여 확인
    # "logp.txt"라는 파일을 만들어서 열기
    # 파일 안에 만들어진 글을 쓰고 저장 (끝에 줄바꿈 추가)
