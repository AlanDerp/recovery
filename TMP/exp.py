# a = [1,2,3,4,5]
# target = 9

# def twoSum(self, nums: List[int], target: int) -> List[int]:
#         temp = sorted(nums)
#         pointerA = 0
#         pointerB = len(nums) - 1
#         while pointerA != pointerB and temp[pointerA]+ temp[pointerB]!= target:
#             if temp[pointerA]+ temp[pointerB] < target:
#                 pointerA += 1
#             if temp[pointerA]+ temp[pointerB] > target:
#                 pointerB -= 1
#         A = temp[pointerA]
#         B = temp[pointerB]
#         pointerA, pointerB = 0,0
#         for i in nums:
#             if i != A:
#                 pointerA += 1
#             else:
#                 break
#         for i in nums:
#             if i != B:
#                 pointerB += 1
#             elif pointerB == pointerA:
#                 pointerB += 1
#             else:
#                 break
#         return pointerA, pointerB


stri = "abcdefghijklmnopqrstuvwxyz"
for a in stri:
    b = ord(a)
    c = 26
    print((b-19) % c)

stringgg = "fhowofhof"
sum = 0
for a in stringgg:
    power = (ord(a) - 96)%26
    sum += 10**power
print(sum)