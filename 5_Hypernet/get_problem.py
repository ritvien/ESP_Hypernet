class Problem():
    # --- Objective Functions ---
    @staticmethod
    def f1(x):
        return 4 * x[0][0]**2 + 4 * x[0][1]**2

    @staticmethod
    def f2(x):
        return (x[0][0] - 5)**2 + (x[0][1] - 5)**2

    def f(self, x):
        return max(self.f1(x), self.f2(x))
    
    # --- Constraint Functions ---
    @staticmethod
    def g1(x):
        return (x[0][0] - 5)**2 + x[0][1]**2 - 25

    @staticmethod
    def g2(x):
        return (x[0][0] - 8)**2 + (x[0][1] + 3)**2 - 17.7

    @staticmethod
    def g3(x):
        return -15 - x[0][0]

    @staticmethod
    def g4(x):
        return x[0][0] - 30

    @staticmethod
    def g5(x):
        return -15 - x[0][1]

    @staticmethod
    def g6(x):
        return x[0][1] - 30

    # Ràng buộc trên không gian mục tiêu  
    @staticmethod
    def g7(x):
        return (Problem.f1(x) - 50)**2 + (Problem.f2(x) - 50)**2 - 50**2
                    