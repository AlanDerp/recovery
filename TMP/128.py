# arraya = [0] * 26
# arraya[25]=1
# arraya += [1]*30
# arraya+=[2]
# print(arraya)
# a,b = 0,0

# # listA = [1,1,1,1,0,0,0,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,1,1]
# # listB = []
# # rear = 0
# # head = 0
# # status = 0
# # listA+=[0]
# # for i in listA:
# #     if status == 0 and i == 0:
# #         rear += 1
# #         head += 1
# #     elif status == 0 and i == 1:
# #         status = 1
# #         rear += 1
# #     elif status == 1 and i == 1:
# #         rear += 1
# #     elif status == 1 and i == 0:
# #         status = 0
# #         listB.append(rear-head)
# #         head = rear

# # print(listB)
        
nums = [1,2,3,4,5]
length = len(nums)
mask = [0]
for i in range(length):
    nums[i] += 1

for num in nums:
    print(num)
    print(int(len(mask)))
    templength = len(mask)
    if num > len(mask):
        print(f"prepare to add {num-templength} zeros")
        mask += [0]*(num+1-templength)
        print("added")
    mask[num-1] = 1
    print(mask)