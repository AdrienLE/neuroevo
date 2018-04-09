import copy
from ga_model import *
class GA:
    def __init__(self, population, compressed_models=None, cuda=False):
        self.population = population
        self.models = [CompressedModel() for _ in range(population)] if compressed_models is None else compressed_models
        self.cuda=cuda

    # Note: the paper says "20k frames", but there are 4 frames per network
    # evaluation, so we cap at 5k evaluations
    def get_best_models(self, env, max_eval=5000):
        results = []
        for m in self.models:
            results.append(evaluate_model(env, m, max_eval=max_eval, cuda=self.cuda))
        used_frames = sum([r[1] for r in results])
        scores = [r[0] for r in results]
        scored_models = list(zip(self.models, scores))
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models, used_frames

    def evolve_iter(self, env, sigma=0.005, truncation=10, max_eval=5000):
        scored_models, used_frames = self.get_best_models(env, max_eval=max_eval )
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]
        scored_models = scored_models[:truncation]
        
        # Elitism
        self.models = [scored_models[0][0]]
        for _ in range(self.population):
            self.models.append(copy.deepcopy(random.choice(scored_models)[0]))
            self.models[-1].evolve(sigma)
            
        return median_score, mean_score, max_score, used_frames

