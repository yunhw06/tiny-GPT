from torch.utils.data import Dataset, DataLoader # PyTorch의 Dataset 클래스 import

class NextTokenDataset(Dataset):  # 다음 토큰 예측용 데이터셋 클래스
    def __init__(self, data, block_size):  
        self.data = data  # 전체 텍스트를 숫자로 바꾼 1차원 텐서
        self.block_size = block_size  # 한 번에 볼 문맥 길이
    def __len__(self):  
        # 만들 수 있는 샘플 수 반환
        # 예: block_size=4이면 [0:4] 입력에 대해 [1:5] 정답이 필요하므로 마지막 일부는 제외
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]  # 입력 시퀀스 x: 현재 위치부터 block_size 길이만큼
        y = self.data[idx + 1 : idx + self.block_size + 1]  # 정답 시퀀스 y: x를 한 칸 오른쪽으로 민 값
        return x, y  # 모델은 x를 보고 y를 예측하도록 학습됨