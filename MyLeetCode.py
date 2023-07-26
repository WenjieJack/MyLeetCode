from typing import List
from collections import Counter
import math


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

    def alphaNum(self, c) -> bool:
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


# P605 Can Place Flowers
class P605:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        f = [0] + flowerbed + [0]

        # skip first and last
        for i in range(1, len(f) - 1):
            if f[i - 1] == 0 and f[i] == 0 and f[i + 1] == 0:
                f[i] = 1
                n -= 1

        return n <= 0


# P28 Find the Index of the First Occurrence in a String
class P28:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == "":
            return 0

        for i in range(len(haystack) + 1 - len(needle)):
            if haystack[i : i + len(needle)] == needle:
                return i

        return -1


# P169 Majority Element
# time and space: O(1)
class P169:
    def majorityElement(self, nums: List[int]) -> int:
        res, count = 0, 0

        for n in nums:
            if count == 0:
                res = n
            count += 1 if n == res else -1
        return res


# P283 Move Zeroes
class P283:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = 0
        for r in range(len(nums)):
            # if it is not 0
            if nums[r]:
                # swap
                nums[l], nums[r] = nums[r], nums[l]
                l += 1


# P724 Find Pivot Index
class P724:
    def pivotIndex(self, nums: List[int]) -> int:
        total = sum(nums)

        leftSum = 0
        for i in range(len(nums)):
            rightSum = total - nums[i] - leftSum
            if leftSum == rightSum:
                return i
            leftSum += nums[i]

        return -1


# P136 Single Number
# XOR, 0^0=0 or 1^1=0, otherwise 1
class P136:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0  # n^0=n
        for n in nums:
            res = n ^ res
        return res


# P448 Find All Numbers Disappeared in an Array
class P448:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        # mark existing
        for n in nums:
            i = abs(n) - 1
            nums[i] = -1 * abs(nums[i])

        res = []
        for i, n in enumerate(nums):
            if n > 0:
                res.append(i + 1)

        return res


# P1189 Maximum Number of Balloons
class P1189:
    def maxNumberOfBalloons(self, text: str) -> int:
        countText = Counter(text)
        ballon = Counter("balloon")

        res = len(text)
        for c in ballon:
            res = min(res, countText[c] // ballon[c])

        return res


# P290 Word Pattern
class P290:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split(" ")
        if len(pattern) != len(words):
            return False

        charToWord, wordToChar = {}, {}

        for c, w in zip(pattern, words):
            if c in charToWord and charToWord[c] != w:
                return False
            if w in wordToChar and wordToChar[w] != c:
                return False
            charToWord[c] = w
            wordToChar[w] = c

        return True


# P682 Baseball Game
class P682:
    def calPoints(self, operations: List[str]) -> int:
        stack = []

        for op in operations:
            if op == "+":
                stack.append(stack[-1] + stack[-2])
            elif op == "D":
                stack.append(2 * stack[-1])
            elif op == "C":
                stack.pop()
            else:
                stack.append(int(op))

        return sum(stack)


# P155 Min Stack
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]


# P15 3Sum
class P15:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        for i, a in enumerate(nums):
            # if a is a value used before
            if i > 0 and a == nums[i - 1]:
                continue

            l, r = i + 1, len(nums) - 1
            while l < r:
                threeSum = a + nums[l] + nums[r]
                if threeSum > 0:
                    r -= 1
                elif threeSum < 0:
                    l += 1
                else:
                    res.append([a, nums[l], nums[r]])
                    l += 1
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1

        return res


# P5 Longest Palindromic Substring
class P5:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        resLen = 0

        for i in range(len(s)):
            # odd length
            l, r = i, i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    res = s[l : r + 1]
                    resLen = r - l + 1
                l -= 1
                r += 1

            # even length
            l, r = i, i + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    res = s[l : r + 1]
                    resLen = r - l + 1
                l -= 1
                r += 1

        return res


# P46 Permutations
class P46:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []

        if len(nums) == 1:
            # nums.copy()
            return [nums[:]]

        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)

            for perm in perms:
                perm.append(n)
            result.extend(perms)
            nums.append(n)

        return result


# P322 Coin Change
# time: O(amount * len(coins)), space: O(amount)
class P322:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # [1,3,4,5]
        # dp[0] = 0, amount = 0, 0 coin
        # dp[1] = 1, amount = 1, 1 coin
        # dp[2] = 0, amount = 2, 2 coin
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - c])

        return dp[amount] if dp[amount] != amount + 1 else -1
