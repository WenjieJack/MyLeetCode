from typing import List, Optional
import math
import heapq


# P242 Valid Anagram
# An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
class P242:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        # hash table
        countS, countT = {}, {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        for c in countS:
            if countS[c] != countT.get(c, 0):
                return False

        return True


# P1 Two Sum
class P1:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        prevMap = {}

        for i, n in enumerate(nums):
            diff = target - n
            if diff in prevMap:
                return [prevMap[diff], i]
            prevMap[n] = i
        return


# P53 Maximum Subarray
class P53:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSub = nums[0]
        curSum = 0

        for n in nums:
            if curSum < 0:
                curSum = 0
            curSum += n
            maxSub = max(maxSub, curSum)

        return maxSub


# P167 Two Sum II - Input Array Is Sorted
class P167:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1

        while l < r:
            curSum = numbers[l] + numbers[r]

            if curSum < target:
                l += 1
            elif curSum > target:
                r -= 1
            else:
                return [l + 1, r + 1]

        return []


# P198 House Robber
class P198:
    def rob(self, nums: List[int]) -> int:
        rob1, rob2 = 0, 0

        # [rob1, rob2, n, n, ...]
        for n in nums:
            temp = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = temp

        return rob2


# P121 Best Time to Buy and Sell Stock
class P121:
    def maxProfit(self, prices: List[int]) -> int:
        l, r = 0, 1  # left is buy, right is sell
        maxP = 0

        while r < len(prices):
            # make profit
            if prices[l] < prices[r]:
                profit = prices[r] - prices[l]
                maxP = max(maxP, profit)
            else:
                l = r
            r += 1

        return maxP


# P88 Merge Sorted Array
class P88:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        # last index nums1
        last = m + n - 1

        # merge in revers order
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[last] = nums1[m - 1]
                m -= 1
            else:
                nums1[last] = nums2[n - 1]
                n -= 1
            last -= 1

        # fill nums1 with leftover nums2 elements
        while n > 0:
            nums1[last] = nums2[n - 1]
            n -= 1
            last -= 1


# P70 Climbing Stairs
class P70:
    def climbStairs(self, n: int) -> int:
        one, two = 1, 1

        for i in range(n - 1):
            temp = one
            one = one + two
            two = temp

        return one


# P20 Valid Parentheses
class P20:
    def isValid(self, s: str) -> bool:
        stack = []
        closeToOpen = {")": "(", "]": "[", "}": "{"}

        for c in s:
            # if c is closing parenthes
            if c in closeToOpen:
                # stack[-1] just added
                if stack and stack[-1] == closeToOpen[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)

        return True if not stack else False


# P7 Reverse Integer
class P7:
    def reverse(self, x: int) -> int:
        # Integer.MIN_VALUE = -2147483648 (end with -8)
        # Integer.MAX_VALUE = 2147483647 (end with 7)

        MIN = -2147483648  # -2^31
        MIN_1 = MIN // 10
        MIN_2 = MIN % 10
        MAX = 2147483647  #  2^31 - 1
        MAX_1 = MAX // 10
        MAX_2 = MAX % 10

        res = 0
        while x:
            digit = int(math.fmod(x, 10))  # (python dumb) -1 %  10 = 9
            x = int(x / 10)  # (python dumb) -1 // 10 = -1

            if res < MIN_1 or (res == MIN_1 and digit <= MIN_2):
                return 0
            if res > MAX_1 or (res == MAX_1 and digit >= MAX_2):
                return 0
            res = res * 10 + digit

        return res


# P35 Search Insert Position
class P35:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # log(n)
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if target == nums[mid]:
                return mid

            if target > nums[mid]:
                l = mid + 1
            else:
                r = mid - 1

        # edge case: target = 1, [2]
        return l


# P26 Remove Duplicates from Sorted Array
class P26:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 1

        for r in range(1, len(nums)):
            # if meet a new unique number
            if nums[r] != nums[r - 1]:
                nums[l] = nums[r]
                l += 1

        return l


# P263 Ugly Number
class P263:
    def isUgly(self, n: int) -> bool:
        if n <= 0:
            return False

        for p in [2, 3, 5]:
            while n % p == 0:
                n = n // p

        # return if n is divisible by 2,3,5
        return n == 1


# P27 Remove Element
class P27:
    def removeElement(self, nums: List[int], val: int) -> int:
        k = 0

        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1

        return k


# P929 Unique Email Addresses
class P929:
    def numUniqueEmails(self, emails: List[str]) -> int:
        unique = set()

        # e is each email address
        for e in emails:
            i = 0
            local = ""  # the local string
            while e[i] not in ["@", "+"]:
                if e[i] != ".":
                    local += e[i]
                i += 1

            while e[i] != "@":
                i += 1

            domain = e[i + 1 :]
            unique.add((local, domain))

        return len(unique)


# P746 Min Cost Climbing Stairs
class P746:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # the top is one more from the array
        cost.append(0)

        for i in range(len(cost) - 3, -1, -1):
            cost[i] += min(cost[i + 1], cost[i + 2])

        return min(cost[0], cost[1])


# P125 Valid Palindrome
class P125:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1

        while l < r:
            while l < r and not self.alphaNum(s[l]):
                l += 1
            while r > l and not self.alphaNum(s[r]):
                r -= 1

            if s[l].lower() != s[r].lower():
                return False
            l, r = l + 1, r - 1

        return True

    def alphaNum(cls, c):
        return (
            (ord("A") <= ord(c) <= ord("Z"))
            or (ord("a") <= ord(c) <= ord("z"))
            or (ord("0") <= ord(c) <= ord("9"))
        )


# P205 Isomorphic Strings
# s = "egg", t = "add" true
class P205:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mapST, mapTS = {}, {}

        for i in range(len(s)):
            c1, c2 = s[i], t[i]

            if (c1 in mapST and mapST[c1] != c2) or (c2 in mapTS and mapTS[c2] != c1):
                return False
            mapST[c1] = c2
            mapTS[c2] = c1

        return True


# P191 Number of 1 Bits
class P191:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n != 0:
            n = n & (n - 1)
            res += 1
        return res


# P217 Contains Duplicate
class P217:
    def containsDuplicate(self, nums: List[int]) -> bool:
        hashset = set()

        for n in nums:
            if n in hashset:
                return True
            hashset.add(n)

        return False


# P703 Kth Largest Element in a Stream
# Heap queue algorithm (a.k.a. priority queue), min-heap
class P703:
    def __init__(self, k: int, nums: List[int]):
        self.minHeap, self.k = nums, k
        heapq.heapify(self.minHeap)
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)

    def add(self, val: int) -> int:
        heapq.heappush(self.minHeap, val)
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        return self.minHeap[0]
