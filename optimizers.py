import numpy as np
import jax.numpy as jnp
import pickle  # 确保导入了 pickle

class BaseOptimizer:
    def ask(self): raise NotImplementedError
    def tell(self, x, fitness): raise NotImplementedError
    def get_pareto_front(self): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError

# --- 将 GenericProblem 移出到全局作用域 ---
from pymoo.core.problem import Problem

class GenericProblem(Problem):
    def __init__(self, num_dims, num_objs):
        super().__init__(n_var=num_dims, n_obj=num_objs, xl=-1.0, xu=1.0)

# --- pymoo 适配器 ---
class PymooOptimizer(BaseOptimizer):
    def __init__(self, algo_name, pop_size, num_dims, num_objs):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        
        # 使用全局定义的类
        self.problem = GenericProblem(num_dims, num_objs)
        self.algorithm = NSGA2(pop_size=pop_size)
        self.algorithm.setup(self.problem, termination=('n_gen', 1000))

    def ask(self):
        self.pop = self.algorithm.ask()
        return jnp.array(self.pop.get("X"))

    def tell(self, x, fitness):
        self.pop.set("F", np.array(fitness))
        self.algorithm.tell(infills=self.pop)

    def get_pareto_front(self):
        res = self.algorithm.result()
        return res.X, res.F
    
    def save(self, path):
        # 现在的 self.algorithm 可以被 pickle 序列化了
        with open(path, "wb") as f:
            pickle.dump(self.algorithm, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.algorithm = pickle.load(f)
