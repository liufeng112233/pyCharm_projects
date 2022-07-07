import random


# 冒泡排序
def bubble_sort(li):
    for i in range(len(li) - 1):  # 定义趟数
        exchange = False  # 添加标志位，主要是解决无序区也自然有序的情况[9,8,7,1,2,4,5,6]
        for j in range(len(li) - i - 1):
            if li[j] < li[j + 1]:  # 降序<，升序>
                li[j], li[j + 1] = li[j + 1], li[j]  # 两个数交换位置
                exchange = True
        if not exchange:
            return


li = [random.randint(0, 10000) for i in range(10)]
print(li)
bubble_sort(li)
print(li)


def select_sort(li):
    li_new = []
    for i in range(len(li)):
        min_val = min(li)
        li_new.append(min_val)
        li.remove(min_val)
    return li_new


li = [3, 2, 4, 1, 5, 6, 7, 9]
print(select_sort(li))


def insert_sort(li):
    for i in range(1, len(li)):  # 表示摸到的牌的下标
        temp = li[i]
        j = i - 1  # j表示的是手里的牌的下标
        while j >= 0 and li[j] > temp:
            li[j + 1] = li[j]
            j -= 1
            li[j + 1] = temp
            print(i)
