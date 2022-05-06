import random

m = 3
n = 4

img = [[0 for j in range(1, n + 1)] for i in range(1, m + 1)]
d = [[random.randint(0, 100) for j in range(1, n + 1)]
     for i in range(1, m + 1)]


def find_least_seam(d, m, n):
    dp = [[0 for j in range(1, n + 1)] for i in range(1, m + 1)]
    print(d)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if i == 1:
                dp[i - 1][j - 1] = d[i - 1][j - 1]
            else:
                valid = float('inf')
                for k in range(-1, 2):
                    if n > k + j >= 0:
                        valid = min(valid, dp[i - 2][k + j - 1])
                dp[i - 1][j - 1] = d[i - 1][j - 1] + valid
    print(dp)
    return min(dp[m - 1])

# print(find_least_seam(d, m, n))

def break_string_cost(string_length: int, cuts: list[int]):
    n = len(cuts)
    m = n + 1
    dp = [[(0, -1) for _ in range(m)] for _ in range(m)]
    for num_break_points in range(1, m):
        for i in range(0, m - num_break_points):
            j = i + num_break_points
            min_cost = (float("inf"), -1)
            for k in range(i, j):
                cost = (dp[i][k][0] + dp[k+1][j][0], k)
                if cost[0] < min_cost[0]:
                    min_cost = cost
            string_start = 0 if i == 0 else cuts[i - 1]
            string_end = string_length if j >= n else cuts[j]
            dp[i][j] = (min_cost[0] + string_end - string_start, min_cost[1])
    cost = dp[0][n][0]
    return cost

print(break_string_cost(20, [2, 8, 10]))
