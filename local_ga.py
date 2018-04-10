import copy
from ga_model import *
from utils import RandomGenerator
class ScoredModel:
    def __init__(self, model,score=None):
        self.model = model
        self.score = score

    def evaluate(self,env, max_eval,cuda, env_seed=None):
        if self.score is None:
            self.score, used_frames = evaluate_model(env,self.model,max_eval=max_eval,
                    cuda=cuda,env_seed=env_seed)
        else:
            used_frames = 0
        return self.score, used_frames

class GA:
    def __init__(self, population, compressed_models=None, cuda=False, seed=None):
        self.population = population
        if seed is not None:
            self.rng = RandomGenerator(rng=seed)
        if compressed_models is None:
            self.models = [ScoredModel(CompressedModel(start_rng=self.rng.generate())) for _ in range(population)]
        else:
            self.models = [ScoredModel(c) for c in compressed_models]
        self.cuda=cuda

    # Note: the paper says "20k frames", but there are 4 frames per network
    # evaluation, so we cap at 5k evaluations
    def get_best_models(self, env, max_eval=5000):
        results = []
        #for m in self.models:
        #    results.append(evaluate_model(env, m, max_eval=max_eval, cuda=self.cuda))
        used_frames = 0
        for m in self.models:
            if self.rng:
                r, frames = m.evaluate(env, max_eval=max_eval, cuda=self.cuda,
                        env_seed=self.rng.generate())
            else:
                r, frames = m.evaluate(env, max_eval=max_eval, cuda=self.cuda)
            used_frames+=frames

        scores = [r.score for r in self.models]
        #scored_models = list(zip(self.models, scores))
        
        self.models.sort(key=lambda x: x.score, reverse=True)
        return used_frames

    def selection(self, scores,k=4.0):
        import numpy as np
        cb, cw = scores[0], scores[-1]
        fitness = [ (ci - cw) + (cb-cw) / (k-1.0) for ci in scores]
        prob = fitness / np.sum(fitness)
        if self.rng:
            rand = self.rng
        else:
            rand = np.rancom
        if np.sum(prob) == 0:
            i = rand.choice(np.arange(len(scores)))
        else:
            i = rand.choice(np.arange(len(scores)),p=prob)
        return i

    def evolve_iter(self, env, sigma=0.005, truncation=5, max_eval=2000):
        used_frames = self.get_best_models(env, max_eval=max_eval)
        scores = [m.score for m in self.models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        i = self.selection(scores)
        parent = self.models[i].model
        child = copy.deepcopy(parent)
        child.evolve(sigma, rng_state=self.rng.generate() )
        del self.models[-1]
        #del parent 
        self.models.append(ScoredModel(child))
        return median_score, mean_score, max_score, used_frames, self.models[0].model
 
    '''
    def evolve_iter(self, env, sigma=0.005, truncation=5, max_eval=5000):
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
    '''

if __name__ == '__main__':
    ga = GA(3)
    for i in range(5):
        _,_,_,frames,m = ga.evolve_iter('FrostbiteNoFrameskip-v4')
        print(frames)
