import sys

sys.stdin = open("sample_input.txt", "r", encoding="utf-8")
T = 10
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T + 1):
    V, E = map(int, input().split())
    print('v,e: ', V, E)
    graph = [[] for _ in range(V + 1)]
    in_degree = [0] * (V + 1)

    lists = list(map(int, input().split()))
    print(lists)