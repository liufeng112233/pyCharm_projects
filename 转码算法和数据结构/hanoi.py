def hanoi(n, a, b, c):  # 汉诺递推公式：h(x)=2h(x-1)+1
    if n > 0:
        hanoi(n - 1, a, c, b)
        print("moving from %s to %s" % (a, c))
        hanoi(n - 1, b, a, c)


hanoi(64, 'A', 'B', 'C')

def linear_search(li, val):  # 时间复杂度O(n)
    for ind, v in enumerate(li):
        if v == val:
            return ind
        else:
            return None


def binary_search(li, val):
    left = 0
    right = len(li) - 1
    while left <= right:  # 说明候选区还有数据
        mid = (left + right) // 2
        if li[mid] == val:
            return mid
        elif li[mid] > val:  # mid的值在左侧
            right = mid - 1
        else:  # li[mid]<val mid的值在右侧
            left = mid + 1
    else:
        return None


li = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(binary_search(li, 3))


from sklearn.metrics import classification_report
import torch