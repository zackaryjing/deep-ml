# Optimal String Alignment Distance

def OSA(source: str, target: str) -> int:
    # Your code here
    n,m = len(source),len(target)
    dp = [[0] * (m + 1)  for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(n + 1):
        dp[i][0] = i
    dp[0][0] = 0
    for j in range(1,m + 1):
        for i in range(1,n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            elif i > 1 and j > 1 and source[i - 1] == target[j - 2] and target[j - 1] == source[i - 2]:
                dp[i][j] = dp[i - 2][j - 2] + 1
            elif source[i - 1] != target[j - 1]:
                dp[i][j] = min(dp[i - 1][j - 1] + 1,dp[i][j - 1] + 1,dp[i - 1][j] + 1)
    return dp[n][m]
            
            


def main():
    source = "butterfly"
    target = "dragonf"

    distance = OSA(source, target)
    print(distance)

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-03 17:38:02
#
