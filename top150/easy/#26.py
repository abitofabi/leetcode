class Solution:
    def removeDuplicates(self, nums):
        if not nums:
            return 0

        k = 1  # Pointer to place the next unique element

        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]
                k += 1

        return k

#alretnate solution

class Solution(object):
    def removeDuplicates(self, nums):
        nums[:]=sorted(set(nums))
        return len(nums)  
