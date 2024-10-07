import heapq

class Solution(object):
    def __init__(self, target='123804765'):
        self.s_target = target

    def H(self, s_cur):
        s_target = self.s_target
        num = 0
        for i in range(9):
            if s_target[i] != s_cur[i] and s_cur[i] != '0': 
                num += 1
        return num

    def Astar(self, s0):
        if self.H(s0) == 0:
            return 0

        heap = []
        visited = set() 
        heapq.heappush(heap, [self.H(s0), s0, s0.index('0'), 0])

        neighbors = {
            0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
            3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
            6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
        }

        ctrl_steps = 0 # control print
        while heap:
            _, cur_s, zero_pos, steps = heapq.heappop(heap)

            if steps == ctrl_steps:
                print(f"步骤 {steps}，当前状态: {cur_s}")
                ctrl_steps += 1

            if cur_s == self.s_target:
                return steps

            if cur_s in visited: 
                continue

            visited.add(cur_s)

            for neighbor in neighbors[zero_pos]:
                new_s = list(cur_s)  # str to list
                new_s[zero_pos], new_s[neighbor] = new_s[neighbor], new_s[zero_pos]  # exchange order
                new_s = ''.join(new_s)  # list to str
                if new_s not in visited: 
                    heapq.heappush(heap, [self.H(new_s) + steps + 1, new_s, neighbor, steps + 1])

        return -1  # error


s0 = '216408753'
solution = Solution()
print(solution.Astar(s0))