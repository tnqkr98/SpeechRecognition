
"""
6개월만에 PyTorch 복습(20/07/10)
"""
import numpy as np
import torch
print(np.__version__)
print(torch.__version__)

t = np.array([0.,1.,2.,3.,4.,5.,6.,])
print(t)
print(t.ndim)   # rank = 1
print(t.shape)  # (7,)  : row 7  col 0.  기본적으로 최상단 , 는 행의 구분.

t = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

print(t)
print(t.ndim)       # rank = 2 (차원이라고 봐도 무방 일단은)
print(t.shape)      # (3,3) : row 3 col 3

# -------------------Torch-------------
t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t)                    # tensor([0., 1., 2., 3., 4., 5., 6.])
print(t.dim())      # 차원 = 1
print(t.shape)      # torch.Size([7])   row :7 ,col :0
print(t.size())     # torch.Size([7])

t = torch.FloatTensor([
    [1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.],
    [10.,11.,12.]
])
print(t)
print(t.dim())      # 차원 = 2
print(t.size())     # torch.size([4,3])     row : 4, col : 3
print(t[:, 1])      # [모든행의, 1번인덱스]

t = torch.FloatTensor([
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        [
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24]
        ]
    ]
])
print(t.dim())      # 차원 = 4
print(t.size())     # shape = torch.size([1,2,3,4])

m1 = torch.FloatTensor([[1, 2], [3, 4]])  # 2 x 2 행렬
m2 = torch.FloatTensor([[1], [2]])        # 2 x 1 행렬

print(m1.matmul(m2))    #  [1,2] X [1] = [5]   행렬곱 수행(크기 맞춰야)
                        #  [3,4]   [2]   [11]
print(m1.mul(m2))       #  [1,2] * [1]  ->  [1,2] *  [1,1] -> [1*1 , 2*1] = [1,2]
print(m1*m2)            #  [3,4]   [2]      [3,4]    [2,2]    [3*2 , 4*2]   [6,8]
                        #                   (BroadCasting)     (요소단위곱 수행)
# BroadCasting
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])
print(m1+m2)                    # [[1,2]] + [[3,3]] = [[4,5]]

m1 = torch.FloatTensor([[1, 2]])        #[[1,2] + [3]] ->  [[1,2] +[3,3]]
m2 = torch.FloatTensor([[3], [4]])      #[[1,2] + [4]] ->  [[1,2] +[4,4]]
print(m1+m2)

# Function
t = torch.FloatTensor([[1, 2], [3, 4]])   # 정수형 텐서는 mean() 불가
print(t.mean())                           #  전체 평균을 스칼라로 (전부 반환형은 텐서)
print(t.mean(dim=0))                      #  열평균을 1차원 벡터로
print(t.mean(dim=1))                      #  행평균을 1차원 벡터로 dim=-1 과 같음
print(t.sum())                            # 전체 합 스칼라
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.max())                            # 전체 중 최대
print(t.max(dim=0))                       # 두개 나옴 0번인덱스 : 열max, 1번: 열argmax
print(t.max(dim=0)[0])                    # 열 MAX
print(t.max(dim=0)[1])                    # 열 argmax : 가장 높은 값의 인덱스
print(t.max(dim=1))                       # 행 Max , argmax

# View (Numpy의 Reshape과 유사)
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
print(ft.view([-1,3]))      #   2차원 텐서로 바꾸고, 열은 3 행은 알아서(-1) 하라.
print(ft.view([-1,3]).shape)  # [2,2,3] 모양이 [4,3] 모양 으로 바뀜.
print(ft.view([-1,1,3]))        # 3차원 텐서로 바꾸고, 열 3, 행1(높이는 알아서 -1) 하라
print(ft.view([-1,1,3]).shape)  # [2,2,3] 모양이 [4,1,3] 모양 으로 바뀜.

ft = torch.Tensor([0, 1, 2])
print(ft.view(1, -1))           # (행은 1 열은 알아서)
print(ft.view(1, -1).shape)     # (1,3) 모양   [[0,1,2]] 가 됨

# Squeeze
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)             # torch.Size([3,1])  모양이었는데,'1'은 사실상 없어도 되는 차원
print(ft.squeeze())         # [0,1,2] 가 됨.
print(ft.squeeze().size())  # torch.Size([3])

# Unsqueeze
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(0))      # 한 단계 높은 차원을 강제 추가
print(ft.unsqueeze(1))      # 한 단계 높은 차원을 추가하고 현재 차원의 요소들을 더 높은차원의 요소들로
print(ft.unsqueeze(0).shape)        # 모양 (3) -> (1,3)
print(ft.unsqueeze(1).shape)        # 모양 (3) -> (3,1)

# Scatter ( for one-hot encoding)
lt = torch.LongTensor([[0], [1], [2], [0]])
print(lt)
one_hot = torch.zeros(4,3)  # 4x3 0으로 채워진 실수텐서 생성  ex ) batch_size=4, classes =3
print(one_hot)
one_hot.scatter_(1,lt,1)    # 의미 검색 요망. 하여간 원핫인코딩됨.
print(one_hot)

# Casting
lt = torch.LongTensor([1,2,3,4])
print(lt)
print(lt.float())
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

# Concatenation
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x,y],dim=0))       # 행으로 연결
print(torch.cat([x,y],dim=1))       # 열로 연결

# Stacking (Concatenaion 의 확장)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))           # 행으로 쌓음(보통 dim=0 이 default)
print(torch.stack([x, y, z], dim=1))    # 열로 쌓음

# Ones and Zeros Like
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x))           # 모양 유지, 모든 값을 1로
print(torch.zeros_like(x))          # 모양 유지, 모든 값을 0로

# In-place Operation
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))                # mul() 은 본래 값을 바꾸지 않음.
print(x)
print(x.mul_(2.))               # mul_() 은 본래 값을 바꾸어버림.
print(x)