from enum import IntEnum
from random import choice, choices, sample

from includes.utils import precision_round
from .base import *
import copy

# ---- set of strategy functions to decide when an item overshoots and what ot do ----    
# TODO define somewhere meaningful

class Overshoot(IntEnum):
    OVERSHOOT_LAST_LEVEL = -1
    OVERSHOOT_INTERMEDIATE = -2

def find_queue_for_item(obj, item):
    """Returns the queue index for an item based on its length and on the demotion thresholds"""
    # TODO: IMPORTANT! Need to be way more efficient here! Segment tree?
    for qid, r in enumerate(obj.queue_allowed_job_size_ranges):
        if (item.obtained_service < r[1]) and (item.obtained_service >= r[0]):
            return qid
    return Overshoot.OVERSHOOT_INTERMEDIATE  # overshoot, send to next server

def find_queue_for_item_last_server_in_hybrid_spatial(obj, item):
    """ Same as find_queue_for_item but makes sure that when the item is about to be demoted to the lowest
    overall priority level, i.e., obtained service equal to last demotion threshold, it is first 
    sent through load balancing step, by returning none match to a priority queue. See on_overshoot_random_choice"""
    # TODO: IMPORTANT! Need to be way more efficient here! Segment tree?
    for qid, r in enumerate(obj.queue_allowed_job_size_ranges):
        if (item.obtained_service < r[1]) and (item.obtained_service >= r[0]):    
            if (qid == len(obj.queue) - 1) and (item.obtained_service == r[0]):
                # Ugly: add epsilon service time to obtained service, otherwise if item is load
                # balanced to the same server, it will not have a queue assigned.
                epsilon = np.finfo(np.float32).eps
                item.update_obtained_service(epsilon)
                return Overshoot.OVERSHOOT_LAST_LEVEL # overshoots, random choice if it is last overshoot, otherwise send to next server
            return qid
    return Overshoot.OVERSHOOT_INTERMEDIATE
    
#-----------------------

class Server(ConnectableItem):
    is_serving = False
    average_per_queue_job_size = {'tot': [], 'n': []}
    idle_time = 0.0
    idle_start_time = 0.0

    def __init__(self, mu, env, server_idx=0, debug=False):
        super().__init__()
        self.mu = mu
        self.env = env
        self.server_idx = server_idx
        self.debug = debug
        
    def log(self, s, end='\n'):
        if self.debug:
            print(f't={self.env.now} -- [Server #{self.server_idx}]: {s}', end=end)

    def attach_queues(self, queue):
        self.queue = queue

    def serve(self):
        
        self.is_serving = True
        
        while True:
            if self.queue.length() == 0:
                break

            # Get a pointer for the first item in the queue
            item = self.queue.get_item_to_serve()
            time_to_serve = self.get_amount_to_serve(item) / self.mu
            serving_time = time_to_serve/self.mu
            yield self.env.timeout(serving_time)  # Serve the item

            # Update the obtained service
            item.update_obtained_service(time_to_serve)

            # Check if the item has completed service and send it to sink
            if item.get_missing_service() == 0:
                item = self.queue.pop()
                self.link.send(self, item)

        self.is_serving = False

    def item_arrived_at_queue(self, queue_idx, item):
        if not self.is_serving:
            self.pid = self.env.process(self.serve())

    def get_amount_to_serve(self, item):
        return item.get_missing_service()

    def receive(self, item):
        raise Exception(
            "Trying to call receive on an server! Receive should be called on the corresponding queue")


class SPServer(Server):
    """Utilizes a list of queues by assuming queue[0] to be the highest priority queue"""
    demotion_index = []  # stores mapping queue_index -> max_job_size_in_queue


    def __init__(self, 
                 mu, 
                 env,
                 server_idx,
                 find_queue_for_item=find_queue_for_item,
                 lb_weights=None,
                 debug=False):
        super().__init__(mu, env, server_idx, debug)
        
        self.queue_allowed_job_size_ranges = []
        # strategy functions
        self.find_queue_for_item = find_queue_for_item # must receive obj and item and return queue
        self.lb_weights = lb_weights    # probability distribution for load balancing

    def serve(self):

        while True:
            # TODO: Keeping a sorted list of non empty queues instead of iterating over all queues is faster!
            queue_to_serve = self.get_queue_to_serve()
            if queue_to_serve == None:
                break
            item = queue_to_serve.pop()  # Get the first item in the queue
            amount_to_serve = self.get_amount_to_serve(item)
            time_to_serve = amount_to_serve / self.mu
            assert time_to_serve > 0
            
            reinsert_to_top = False
            self.packet_arrived = self.env.event()

            start_serving_time = self.env.now
            yield (self.env.timeout(time_to_serve)  | self.packet_arrived) # Serve the item
            
            # if timeout event: we have served exactly the amount we wanted,
            # otherwise we've been preempted by a new packet and we need to compute the amount served
            if self.packet_arrived.triggered:
                self.log(f'service preempted, arrived new packet {self.packet_arrived.value}')
                served = (self.env.now - start_serving_time) * self.mu
                reinsert_to_top = True # if preempted by packet need to reinsert to top and continue serving in FIFO
            else:
                self.log(f'service finished, served {amount_to_serve} in {time_to_serve} seconds')
                served = amount_to_serve

            # Update the obtained service
            item.update_obtained_service(served)

            # Check if the item has completed service and send it to sink or reinsert it
            overshoot, reason = self.item_size_overshoots_queue(item)
            if item.get_missing_service() <= 0:  # Send to sink
                self.link[0].send(self, item)
            elif overshoot:
                self.on_overshoot(item, reason)
            else: 
                self.reinsert_in_queue(item, reinsert_to_top)

        self.is_serving = False
        self.idle_start_time = self.env.now
        self.log(f'going idle at {self.idle_start_time}')

    def attach_queues(self, queues, job_size_ranges):
        self.queue = queues
        if len(self.queue) != len(job_size_ranges):
            raise AttributeError("Number of job size intervals doesn't match number of queues")
        
        print(f'Server #{self.server_idx}: [', end='')
        for i in range(len(job_size_ranges)):
            self.queue_allowed_job_size_ranges.append(job_size_ranges[i])  # maps queue -> allowed job size range
            print(job_size_ranges[i], end='')
        print(']')

    def get_queue_size(self):
        return [q.length() for q in self.queue]

    def reinsert_in_queue(self, item, reinsert_to_top=False):
        qid = self.find_queue_for_item(self, item)
        q = self.queue[qid]
        assert q is not None
        
        if reinsert_to_top:
            q.reinsert(0, item)
        else:
            q.reinsert(q.length(), item)
        return q
    
    def get_amount_to_serve(self, item):
        """ Returns the time to serve as the minimum between the next demotion threshold and the missing service time """
        qid = self.find_queue_for_item(self, item)
        max_service_for_queue = self.queue_allowed_job_size_ranges[qid][1]
        return min([item.get_missing_service(), max_service_for_queue - item.get_obtained_service()])

    def item_size_overshoots_queue(self, item):
        queue = self.find_queue_for_item(self, item) 
        return queue < 0, queue
        #print("Queue index: {}, Demotion Index: {}, Job: {}".format(idx, self.demotion_index, item.describe(synthetic=True)))
    
    def on_overshoot(self, item, reason, serving_queue):
        """ Send item to next server in cascade, or if it is the last overshooting step, send it to a random server """

        if reason == Overshoot.OVERSHOOT_INTERMEDIATE:
            self.log(f'Server {self.server_idx} overshoots for id: {item.id}, obtained service: {item.obtained_service}, missing service: {item.get_missing_service()}')
            serving_queue.link[0].send(serving_queue, item)
        elif reason == Overshoot.OVERSHOOT_LAST_LEVEL:
            #out = np.random.choice(serving_queue.link, 1, p=self.lb_weights)
            #out = choice(serving_queue.link)
            out = choices(serving_queue.link, weights=self.lb_weights)[0]
            self.log(f'overshoots to server #{out.b.server.server_idx}:{out.b.queue_idx}'
                f' for id: {item.id}, obtained service: {item.obtained_service}, missing service: {item.get_missing_service()}')
            out.send(serving_queue, item)
        else:
            raise Exception('Invalid overshoot reason')

    def item_arrived_at_queue(self, queue_idx, item):
        """Called by the queue in order to notify the server that the queue state has changed. 
            Interrupts the serving and performs preemption if needed. 
            TODO probably more efficient solution instead of starting a new process would be
            to wait for packet_arrived() event outside of infinite while loop in serve(), then this
            function would just be a trigger for the event."""
        self.log(f'New job: {item.describe(synthetic=True)} arrived at s{self.server_idx}::q{queue_idx}')
        if not self.is_serving:
            self.is_serving = True
            self.idle_time += (self.env.now - self.idle_start_time)
            self.log(f'resuming from idle after {(self.env.now - self.idle_start_time)}, total idle time: {self.idle_time}')
            self.pid = self.env.process(self.serve())
        else:
            if not self.packet_arrived.triggered: # if two jobs arrive back-to-back event has already been triggered
                self.packet_arrived.succeed(f'{item.describe(synthetic=True)}')

    def get_queue_to_serve(self):
        queue_to_serve = None
        for i in range(len(self.queue)):
            if self.queue[i].length() > 0:
                queue_to_serve = self.queue[i]
                break
        return queue_to_serve


class SPServer_PS(SPServer):
    """Utilizes a list of queues by assuming queue[0] to be the highest priority queue"""
    
    def serve(self):
        
        while True:
            
            queue_to_serve = self.get_queue_to_serve()
            if queue_to_serve == None:
                # move to idle state
                break
            
            items = queue_to_serve.pop_all()  # Get all items in the queue
            
            amount_per_item_to_serve = self.get_amount_to_serve(items, queue_to_serve)
            serving_time = amount_per_item_to_serve / self.mu * len(items)  # ideally if not interrupted
            
            self.log(f'serving {[x.id for x in items]} in queue {queue_to_serve.queue_idx}, will serve {amount_per_item_to_serve}')

            assert serving_time > 0
            
            # Serve the job until either finish or new packet arrive, in which case we should re-compute queue to serve
            self.packet_arrived = self.env.event()
            start_serving_time = self.env.now
            
            yield (self.env.timeout(serving_time) | self.packet_arrived)
            
            if self.packet_arrived.triggered:
                self.log(f'service preempted, arrived new packet {self.packet_arrived.value}')
                amount_per_item_served = (self.env.now - start_serving_time) * self.mu / len(items)
            else:
                self.log(f'service finished, served {amount_per_item_to_serve} in {serving_time} seconds')
                amount_per_item_served = amount_per_item_to_serve
            
            # Update the obtained service
            [it.update_obtained_service(amount_per_item_served) for it in items]

            for item in items:
                
                if item.get_missing_service() <= 0:         # Send to sink
                    self.log(f'item {item.describe(synthetic=True)} completed, sending to sink')
                    self.link[-1].send(self, item)          # Note: sink is alawys the link with the last index                
                    continue

                # need demotion or has to be re-enqueued to same queue
                overshoot, reason = self.item_size_overshoots_queue(item)
                if overshoot:   # Send to next processing step
                    self.on_overshoot(item, reason, queue_to_serve)
                else:                                       # Otherwise put it again in a queue
                    q = self.reinsert_in_queue(item, True)
                    self.log(f'item {item.describe(synthetic=True)} re-enqueued to q{q.queue_idx}')

        self.is_serving = False
        self.idle_start_time = self.env.now


    def get_amount_to_serve(self, items, queue):
        """ Returns the time to serve as the minimum between the time left to complete the job 
        and the time left to reach the maximum allowed service for the queue. We consider the shortest
        job in the queue for computations. """
        
        # get maximum service in queue
        max_service_for_queue = self.queue_allowed_job_size_ranges[queue.queue_idx][1]
        
        return min(
                [
                    min([it.get_missing_service() for it in items]),
                    max_service_for_queue - max([it.get_obtained_service() for it in items])
                ]
            )


class Queue(ConnectableItem):

    def __init__(self, env, queue_idx=0):
        super().__init__()
        self.env = env
        self.q = []
        self.queue_idx = queue_idx

    def get_item_to_serve(self):
        item = self.get_front_item()
        return item

    def get_front_item(self):
        if self.length() == 0:
            raise Exception('Trying to peek on an empty queue')
        return self.q[0]

    def push(self, item):
        self.q.append(item)
        self.server.item_arrived_at_queue(self.queue_idx, item)

    def pop(self):
        if self.length() == 0:
            raise Exception('Trying to pop from an empty queue')
        return self.q.pop(0)

    def pop_all(self):
        """ Returns a copy of the queue and empties the current queue """
        _q = copy.deepcopy(self.q)
        self.q.clear()
        return _q

    def reinsert(self, idx, item):
        self.q.insert(idx, item)

    def length(self):
        return len(self.q)

    def attach_server(self, server):
        self.server = server

    def receive(self, item):
        self.push(item)
