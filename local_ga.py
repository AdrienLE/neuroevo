import copy
from ga_model import *
from utils import RandomGenerator

class GA:
    def __init__(self, population, compressed_models=None, cuda=False, seed=None):
        self.population = population
        self.cuda=cuda
        if seed is not None:
            self.rng = RandomGenerator(rng=seed)
        if self.rng:
            self.models = [CompressedModel(start_rng=self.rng.generate()) for _ in range(population)] if compressed_models is None else compressed_models
        else:
            self.models = [CompressedModel() for _ in range(population)] if compressed_models is None else compressed_models
    # Note: the paper says "20k frames", but there are 4 frames per network
    # evaluation, so we cap at 5k evaluations
    def get_best_models(self, env, max_eval=1000):
        results = []
        for m in self.models:
            if self.rng:
                results.append(evaluate_model(env, m, max_eval=max_eval, cuda=self.cuda,
                    env_seed=self.rng.generate()))
            else:
                results.append(evaluate_model(env, m, max_eval=max_eval, cuda=self.cuda))
        used_frames = sum([r[1] for r in results])
        scores = [r[0] for r in results]
        scored_models = list(zip(self.models, scores))
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models, used_frames

    def evolve_iter(self, env, sigma=0.005, truncation=5, max_eval=1000):
        scored_models, used_frames = self.get_best_models(env, max_eval=max_eval )
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]
        scored_models = scored_models[:truncation]
        
        # Elitism
        self.models = [scored_models[0][0]]
        for _ in range(self.population - 1):
            if self.rng:
                self.models.append(copy.deepcopy(self.rng.choice(scored_models)[0]))
                self.models[-1].evolve(sigma, self.rng.generate())
            else:
                self.models.append(copy.deepcopy(random.choice(scored_models)[0]))
                self.models[-1].evolve(sigma)
            
        return median_score, mean_score, max_score, used_frames, scored_models[0][0]

