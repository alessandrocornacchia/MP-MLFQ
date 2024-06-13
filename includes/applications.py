from .base import *

class Application(ConnectableItem):
    def __init__(self, cdf, _lambda, env):
        super().__init__()
        self._lambda = _lambda
        self.cdf = cdf
        self.env = env
        self.generated_jobs = 0
        self.pid = self.env.process(self.start())

    def start(self):
        """ Continuously generates jobs according to the specified distribution """
        while True:
            next_arrival = np.random.exponential(1.0/self._lambda)
            yield self.env.timeout(next_arrival)
            self.generate_job()
        
    def generate_job(self):
        job_size = np.ceil(self.cdf.Fi(random()))
        j = Job(job_size)
        j.set_generation_time(self.env.now)
        j.set_job_id(self.generated_jobs)
        self.generated_jobs += 1
        self.link[0].send(self, j)
        return j

    def receive(self, item):
        raise Exception("Trying to call receive on Application")
        
class Sink(ConnectableItem):
    data = []

    def __init__(self, env, mu):
        super().__init__()
        self.env = env
        self.mu = mu

    def receive(self, item: Job):
        """ Processes a completed job. Saves statistics for later visualization"""
        item.set_completion_time(self.env.now)
        if item.get_missing_service() > 0:
            raise Exception("Sink received an uncompleted job")

        self.data.append(item)
    
   
    def finalize(self):
        """Describes all received items and returns a dictionary of the received data"""
        detailed_data = {}
        for k in self.data[0].describe().keys():
            detailed_data[k] = []
        
        for item in self.data:
            for k in item.describe().keys():
                detailed_data[k].append(item.describe()[k])
            if 'idx' in detailed_data:
                detailed_data['idx'].append(len(detailed_data))
            else:
                detailed_data['idx'] = [0]

        return detailed_data
