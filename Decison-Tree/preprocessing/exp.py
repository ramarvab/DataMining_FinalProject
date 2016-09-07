
def chunkIt(seq, num):
    arr = [len(seq) / num] * num
    k = len(seq) - sum(arr)
    while k > 0:
        arr[k - 1] += 1
        k -= 1
    arr = [i for i in arr if i > 0]
    arr = [0] + arr
    for i in range(1, len(arr)):
        arr[i] += arr[i - 1]
    out = []
    for i in range(0, len(arr) - 1):
        out.append(seq[arr[i]:arr[i + 1]])

    dict = {}
    for part in range(len(out)):
        for point in out[part]:
            dict[point] = part
    return dict

a = range(2)
parts = 10
pts = chunkIt(a, parts)
print pts
'''
def chunkIt(seq, num):
    arr = [len(seq)/num]*num
    k = len(seq) - sum(arr)
    while k > 0:
        arr[k-1] += 1
        k -= 1
    arr = [i for i in arr if i > 0]
    arr = [0] + arr
    for i in range(1,len(arr)):
        arr[i] += arr[i-1]
    out = []
    for i in range(0, len(arr)-1):
        out.append(seq[arr[i]:arr[i+1]])
    print out

a = range(4)
print chunkIt(a, 10)
'''