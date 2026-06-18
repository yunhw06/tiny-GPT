import torch  # PyTorch import
from torch.utils.data import Dataset, DataLoader  # DataLoader import
from dataset import NextTokenDataset  # 사용자 정의 Dataset import

def load_dataloader(text_path, block_size, batch_size = 64):
    text = open(text_path, "r", encoding="utf-8").read()  # 텍스트 파일 전체를 읽어 문자열로 저장
    chars = sorted(list(set(text)))  # 텍스트에 등장하는 고유 문자들을 정렬해 vocab 생성
    stoi = {ch: i for i, ch in enumerate(chars)}  # 문자 -> 정수 인덱스 사전
    itos = {i: ch for ch, i in stoi.items()}  # 정수 인덱스 -> 문자 사전

    vocab_size = len(chars)  # vocab 크기
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)  # 전체 텍스트를 숫자 시퀀스로 변환
    block_size = block_size # 64  
    dataset = NextTokenDataset(data, block_size)  # next-token prediction용 데이터셋 생성

    loader = DataLoader(dataset, batch_size, shuffle=True)  # 데이터셋을 미니배치로 불러오는 DataLoader 생성

    # 모델과 추론에 필요한 정보 저장
    info = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "stoi": stoi,
        "itos": itos,
    }

    return loader, info  # DataLoader와 메타정보 반환

