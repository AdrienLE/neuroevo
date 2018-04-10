import random

class RandomGenerator:
    def __init__(self, rng, seq_len=10000000):
        self.max_int = 2**31-1
        self.rng = rng
        random.seed(rng)
        self.sequence = []
        for i in range(seq_len):
            self.sequence.append(random.randint(0, self.max_int))
        self.current = 0

    def generate(self):
        self.current += 1
        return self.sequence[self.current-1]

    def choice(self, target_list, p=None):
        if p is None:
            return target_list[self.randint(len(target_list))]
        num = float(self.generate()) / self.max_int
        c = 0.0
        for i , pr in enumerate(p):
            c += pr
            if c>= num:
                return target_list[i]
        return target_list[-1]


    def randint(self, n):
        i = self.generate()
        while i >= (self.max_int // n) * n:
            i = self.generate()
        return i % n






