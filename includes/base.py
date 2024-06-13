from random import random, randint
import numpy as np
import simpy


class ConnectableItem(object):
    def __init__(self):
        self.link = []
    
    def receive(self, item):
        raise Exception("Item is a ConnectableItem but it does not implement the receive() method")

    def attach_link(self, link):
        self.link.append(link)


class TopologyManager(object):
    def __init__(self):
        pass

    def attach_link(self, a: ConnectableItem, b: ConnectableItem, direction='both'):
        l = Link(a, b)
        a.attach_link(l) 
        if direction == 'both':
            b.attach_link(l)
        #print("Connecting", a, b)

class Job(object):
    def __init__(self, size, id=0):
        self.size = size
        self.obtained_service=0
        self.id=0
    
    def set_job_id(self, id):
        self.id = id

    def get_job_id(self):
        return self.id

    def get_obtained_service(self):
        return self.obtained_service

    def get_missing_service(self):
        return self.size - self.obtained_service

    def update_obtained_service(self, amount):
        self.obtained_service += amount

    def set_generation_time(self, time):
        self.generation_time = time
    
    def set_completion_time(self, time):
        self.completion_time = time
    
    def get_completion_time(self):
        return self.completion_time
    
    def get_generation_time(self):
        return self.generation_time
    
    def get_job_size(self):
        return self.size

    def describe(self, synthetic=False):
        """ Returns the state of the job. """
        r = {
            'len': self.size, 
            'obtained_serivce': self.obtained_service, 
            'id': self.id
            }
        if not synthetic:
            try:
                r['generation_time'] = self.generation_time
                r['completion_time'] = self.completion_time
            except AttributeError:
                raise Exception('Job appears to have missing generation or completion time. If you are calling this method on an uncompleted job specify synthetic=True')
        return r


class Link(object):
    def __init__(self, s: ConnectableItem, d: ConnectableItem):
        self.a = s
        self.b = d
    
    def send(self, src: ConnectableItem, item: Job):
        if src == self.a:
            self.b.receive(item)
        elif src == self.b:
            self.a.receive(item)
        else:
            raise Exception("Trying to send from a source not attached to this link")