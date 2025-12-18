from typing import Any, Optional, Tuple
from collections import defaultdict
import heapq
import itertools
import random

DASHER_ARRIVAL = "DASHER_ARRIVAL"
TASK_ARRIVAL = "TASK_ARRIVAL"
SIMULATION_END = "SIMULATION_END"
BATCH_DISPATCH = "BATCH_DISPATCH"

class WeightedGraph:
    '''A directed weighted graph using an adjacency list.'''
    def __init__(self):
        ''' Initializer for the graph'''
        self.vertices = []
        self.arcs = []
    
    
    def addNode(self, node):
        '''Add a Vertex to the graph'''
        if node not in self.vertices:
            self.vertices.append(node)
            self.arcs.append([])
        else:
            raise Exception("Node exists")

    def addEdge(self, node1, node2, weight):
        '''Add an edge/arc to the graph'''
        if node1 in self.vertices and node2 in self.vertices:
            index_s = self.vertices.index(node1)
            if not any(neigh == node2 for neigh, _ in self.arcs[index_s]):
                self.arcs[index_s].append((node2, weight))
            
        else:
            print(f"Skipping invalid edge: {node1} → {node2} (weight={weight})")
            return

    def modifyWeight(self, node1, node2, weight):
        '''Modify edge weight'''
        if node1 in self.vertices and node2 in self.vertices:
            index_s = self.vertices.index(node1)
            for i, (neigh, w) in enumerate(self.arcs[index_s]):
                if neigh == node2:
                    self.arcs[index_s][i] = (node2, weight)  
                    return
            raise Exception("Edge does not exist")
        else:
            raise Exception("Not a valid edge")

    def getNeighbors(self, node):
        '''Returns a list of neightbors '''
        if node in self.vertices:
            index_s = self.vertices.index(node)
            return self.arcs[index_s]
        else: 
            raise Exception("Not a valid node")

    def getNodes(self):
        '''Returns the nodes '''
        return self.vertices
    
    def dijkstra_shortest_path(self, start_node, end_node):
        '''Implements Dijkstra's Algorithm to find shortest path'''
        #check case if nodes don't exist
        if start_node not in self.vertices or end_node not in self.vertices:
             return ([], float('inf'))

        #check case if start node is end node
        if start_node == end_node:
            return([start_node], 0)

        #initialize dict of distances and set everything to infinity
        distances = {node: float('inf') for node in self.vertices}

        #set start node distance to self as 0
        distances[start_node] = 0

        #initialize dict of prev nodes to store path
        # Predecessors: Stores the parent node for path reconstruction.
        prev = {node: None for node in self.vertices}

        #priority queue to store the shortest distances, add start_node
        pq = PriorityQueue()
        pq.insert(start_node, 0)

        #dijkstra
        while not pq.isEmpty():
            #get node with min distance
            curr_node, curr_distance = pq.extractMin()

            #if current distance is greater than the shortest one stored, don't change anything
            if curr_distance > distances[curr_node]:
                continue
            
            #if the current node is the end node, then we're done
            if curr_node == end_node:
                break

            #otherwise, check neighbors
            for neighbor, weight in self.getNeighbors(curr_node):
                distance_through_curr = curr_distance + weight
                
                #is this distace better than current shortest
                if distance_through_curr < distances[neighbor]:
                    # then change to this one and update prev
                    distances[neighbor] = distance_through_curr
                    prev[neighbor] = curr_node

                    #decrease key if in priority queue OR add in if not
                    try:
                        pq.decreaseKey(neighbor, distance_through_curr)
                    except Exception:
                        pq.insert(neighbor, distance_through_curr)
                    
                    #go back through loop
        
        #final cost is stored at end_node
        final_cost = distances[end_node]

        #if still equals infinity, then no path found
        if final_cost == float('inf'):
            return ([], float('inf'))

        path = []
        curr = end_node
        
        #go back through prev to retrace steps
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
            
        # reverse path for correct order 
        best_path = path[::-1]
        
        return (best_path, final_cost)
    
    def get_weight(self, node1, node2):
        '''returns the weight of the edge from node1 to node2'''
        if node1 in self.vertices:
            for neighbor, weight in self.getNeighbors(node1):
                if neighbor == node2:
                    return weight
        return None


class PriorityQueue:
    '''A Priority queue implemented using a heap'''
    def __init__(self):
        '''Initializes priority queue. '''
        self.pq = []
    
    def heapifyUp(self, nodeIndex):
        '''Move an item up the heap'''
        if nodeIndex > 0 :
            parentIndex = (nodeIndex-1)//2
            if (self.pq[parentIndex][1] > self.pq[nodeIndex][1]):
                self.pq[parentIndex], self.pq[nodeIndex] = self.pq[nodeIndex], self.pq[parentIndex]
                self.heapifyUp(parentIndex)

    def heapifyDown(self, nodeIndex):
        '''Move an item down the heap'''
        #Stop if node has no children
        if 2*nodeIndex + 1 >= len(self.pq):
            return
        
        #Get index of left child 
        leftChildIndex = (nodeIndex*2) + 1

        ##Get index of right child
        rightChildIndex = (nodeIndex*2) + 2
        
        ##Assume left child is smaller
        smallerChildIndex = leftChildIndex

        # If right child exists, choose the smaller of the two
        if rightChildIndex < len(self.pq) and self.pq[rightChildIndex][1] < self.pq[leftChildIndex][1]:
            smallerChildIndex = rightChildIndex
        
        #Now, check if a swap needs to be done
        if self.pq[nodeIndex][1] > self.pq[smallerChildIndex][1]:
            self.pq[nodeIndex], self.pq[smallerChildIndex] = self.pq[smallerChildIndex], self.pq[nodeIndex]
            self.heapifyDown(smallerChildIndex)


    def insert(self, item, priority):
        '''Adds an element to the heap'''
        self.pq.append((item, priority))
        self.heapifyUp(len(self.pq)-1)

    def extractMin(self):
        '''Removes and returns the minimum'''
        if len(self.pq) == 0:
            return None
        minNode = self.pq[0]
        # Move the last element to the root and shrink the heap
        lastNode = self.pq.pop()
        if len(self.pq) > 0:
            self.pq[0] = lastNode
            self.heapifyDown(0)
        return minNode

    def decreaseKey(self, item, key):
        '''Changes item key value and moves to correct position'''
        index = -1
        for i, (val, priority) in enumerate(self.pq):
            if val == item:
                index = i
                break
    
        if index == -1:
            raise Exception(f"'{item}' not found in the priority queue.")
        self.pq[index] = (item, key)
        self.heapifyUp(index)
    
    def isEmpty(self):
        '''Returns true if the list is empty, false if not'''
        return len(self.pq) == 0


def read_graph(fname):
    # Open the file
    file = open(fname, "r")
    # Read the first line that contains the number of vertices
    numVertices = file.readline()

    # You might need to add some code here to set up your graph object
    graph = WeightedGraph()
    numVertices = int(numVertices)
    for i in range(numVertices):
        graph.addNode(i)

    # Next, read the edges and build the graph
    for line in file:
        # edge is a list of 3 indices representing a pair of adjacent vertices and the weight
        # edge[0] contains the first vertex (index between 0 and numVertices-1)
        # edge[1] contains the second vertex (index between 0 and numVertices-1)
        # edge[2] contains the weight of the edge (a positive integer)
        edge = line.strip().split(",")

    # Use the edge information to populate your graph object
    # TODO: Add your code here
        if len(edge) != 3:
            raise Exception(f"Invalid edge line: {line.strip()}")

        node1 = int(edge[0].strip())
        node2 = int(edge[1].strip())
        weight = float(edge[2].strip())

        print(node1, node2, weight)

        graph.addEdge(node1, node2, weight)
    
    # Close the file safely after done reading
    file.close()
    return graph

def read_dashers(fname):
    # Open the file
    file = open(fname, "r")
    # Set up your list of agents
    dashers=[]

    file.readline()

    # Next, read the agents and build the list
    for line in file:
        # agent is a list of 2 indices representing a pair of vertices
        # path[0] contains the start location (index between 0 and numVertices-1)
        # path[1] contains the destination location (index between 0 and numVertices-1)
        data = line.strip().split(",")
        dasher = {
            'start_location': int(data[0]),
            'start_time': int(data[1]),
            'exit_time': int(data[2])
        }
        dashers.append(dasher)
    
    # Close the file safely after done reading
    file.close()
    return dashers

def read_tasks(fname, appear_time_fixed):
    file = open(fname, "r")
    tasks = []

    file.readline()

    task_id = 0
    for line in file:
        data = line.strip().split(",")
        reward = random.randint(1, 100)
        target_time = int(data[3])
        appear_time = target_time - appear_time_fixed
        task = {
            'task_id': task_id,
            'location': int(data[1]),
            'appear_time': appear_time,
            'target_time': target_time,
            'reward': reward
        }
        tasks.append(task)
        task_id += 1

    file.close()
    return tasks

"""
simple_discrete_event_sim.py

A minimal discrete-event simulator where scheduled events carry an
event_id and optional payload. Event behavior is implemented by
overriding the Simulator.handle(event_id, payload) method using a
simple switch (if/elif) inside it.
"""

class EventHandle:
    """Simple cancelable handle for a scheduled event."""
    __slots__ = ("_cancelled",)
    def __init__(self) -> None:
        self._cancelled = False
    def cancel(self) -> None:
        self._cancelled = True
    @property
    def cancelled(self) -> bool:
        return self._cancelled

#should be fine as is
class Simulator:
    """Minimal discrete-event simulator.
    Subclass and override handle(event_id, payload) with a switch-case.
    """
    def __init__(self, start_time: float = 0.0) -> None:
        self.now = float(start_time)
        self._queue: list[Tuple[float, int, Any, Any, EventHandle]] = []
        self._seq = itertools.count()
        self._stopped = False
        self.events_processed = 0

    def schedule_at(self, time: float, event_id: Any, payload: Any = None) -> EventHandle:
        if time < self.now:
            raise ValueError("Cannot schedule in the past")
        seq = next(self._seq)
        h = EventHandle()
        heapq.heappush(self._queue, (float(time), seq, event_id, payload, h))
        return h

    def _pop_next(self):
        while self._queue:
            time, seq, event_id, payload, h = heapq.heappop(self._queue)
            if not h.cancelled:
                return time, event_id, payload
            # skipped cancelled
        return None

    def step(self) -> bool:
        if self._stopped:
            return False
        item = self._pop_next()
        if item is None:
            return False
        time, event_id, payload = item
        self.now = time
        # dispatch to user-defined handler
        self.handle(event_id, payload)
        self.events_processed += 1
        return True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> None:
        self._stopped = False
        processed = 0
        while not self._stopped:
            if not self._queue:
                break
            if until is not None and self._queue[0][0] > until:
                break
            if max_events is not None and processed >= max_events:
                break
            if not self.step():
                break
            processed += 1

    def stop(self) -> None:
        self._stopped = True

    def handle(self, event_id: Any, payload: Any) -> None:
        """Override in a subclass with a simple switch (if/elif) on event_id."""
        raise NotImplementedError("Override handle(event_id, payload)")

class Baseline(Simulator):
    def __init__(self, graph: WeightedGraph, dashers, tasks):
        super().__init__()
        self.graph = graph
        self.dashers_data = dashers
        self.tasks_data = tasks
        #initializing the simulation state
        self.available_tasks = []
        self.dashers = {}
        self.total_score = 0.0
        self.num_dashers = len(dashers)
        self.dashers_exited = 0

    #access tasklog here
    def start_simulation(self):
        """Initializes the simulation by scheduling initial events."""
        # schedule all dashers' arrivals at their start locations
        for i, dasher_info in enumerate(self.dashers_data):
            dasher_id = i
            self.dashers[dasher_id] = {
                'location': dasher_info['start_location'],
                'exit_time': dasher_info['exit_time'],
                'status': 'available', #either 'available' or 'unavailable'
                'current_task': None
            }

            #schedule arrival in the system
            payload = {
                'dasher_id': dasher_id,
                'location': dasher_info['start_location']
            }
            self.schedule_at(dasher_info['start_time'], DASHER_ARRIVAL, payload)
        
        # schedule all task arrivals
        for task in self.tasks_data:
            payload = {
                'task': task
            }
            self.schedule_at(task['appear_time'], TASK_ARRIVAL, payload)

    def handle(self, event_id: Any, payload: Any):
        if event_id == TASK_ARRIVAL:
            self.handle_task_arrival(payload)
            
        elif event_id == DASHER_ARRIVAL:
            self.handle_dasher_arrival(payload)
            
        elif event_id == SIMULATION_END: 
            self.handle_simulation_end(payload)

    # probably less relevant because "requests" go in the opposite direction now
    # -> yes this code ended up being much shorter (car request got repurposed into task arrival)
    def handle_task_arrival(self, payload: Any):
        task = payload['task']
        self.available_tasks.append(task)        

    # can probably reuse to an extent
    # -> actually this didn't end up resembling handle_car_arrival that much
    def handle_dasher_arrival(self, payload: Any):
        dasher_id = payload['dasher_id']
        location = payload['location']

        # If dasher arrives at a task location, complete task
        current_task = self.dashers[dasher_id].get('current_task')
        if current_task is not None and location == current_task['location']:
            self.total_score += current_task['reward'] #only updates the total_score if the dasher completes the task
            self.dashers[dasher_id]['current_task'] = None

        #update dasher location and status
        self.dashers[dasher_id]['location'] = location
        self.dashers[dasher_id]['status'] = 'available'

        # check if dasher has exited yet
        if self.now >= self.dashers[dasher_id]['exit_time']:
            self.dashers_exited += 1
            if self.dashers_exited == self.num_dashers:
                self.schedule_at(self.now, SIMULATION_END, None)
            return

        #find closest feasible/available task
        closest_task = None
        min_travel_time = float('inf')
        for task in self.available_tasks:
            path, travel_time = self.graph.dijkstra_shortest_path(
                location, task['location']
            )

            if path:
                arrival_time = self.now + travel_time

                if arrival_time <= task['target_time'] and arrival_time <= self.dashers[dasher_id]['exit_time']:
                    if travel_time < min_travel_time:
                        min_travel_time = travel_time
                        closest_task = task

        if closest_task is not None:
            self.dashers[dasher_id]['status'] = 'unavailable'
            self.dashers[dasher_id]['current_task'] = closest_task
            self.available_tasks.remove(closest_task)

            next_payload = {
                'dasher_id': dasher_id,
                'location': closest_task['location']
            }

            # schedule arrival at the task location after travel time
            self.schedule_at(
                self.now + min_travel_time,
                DASHER_ARRIVAL,
                next_payload
            )

    # actually didn't need a check_for_simulation_end since it has a set end
    # handle_simulation_end can be exactly the same
    def handle_simulation_end(self, payload: Any):
        """
        Stops the simulation.
        """
        print("SIMULATION_END event received. Stopping run loop.")
        self.stop()

    def get_total_score(self):
        """ return the total system score """
        return self.total_score

class OpportunityCost(Simulator):
    """Opportunity Cost policy simulator class"""
    def __init__(self, graph: WeightedGraph, dashers, tasks):
        super().__init__()
        self.graph = graph
        self.dashers_data = dashers
        self.tasks_data = tasks
        # initializing the simulation state
        self.available_tasks = []
        self.dashers = {}
        self.total_score = 0.0
        self.num_dashers = len(dashers)
        self.dashers_exited = 0

    # access tasklog here
    def start_simulation(self):
        """Initializes the simulation by scheduling initial events."""
        # schedule all dashers' arrivals at their start locations
        for i, dasher_info in enumerate(self.dashers_data):
            dasher_id = i
            self.dashers[dasher_id] = {
                'location': dasher_info['start_location'],
                'exit_time': dasher_info['exit_time'],
                'status': 'available',  # either 'available' or 'unavailable'
                'current_goal': None,  # current task at hand
                'goal_expiration': None  # target time of current task
            }

            # schedule arrival in the system
            payload = {
                'dasher_id': dasher_id,
                'location': dasher_info['start_location']
            }
            self.schedule_at(dasher_info['start_time'], DASHER_ARRIVAL, payload)

        # schedule all task arrivals
        for task in self.tasks_data:
            payload = {
                'task': task
            }
            self.schedule_at(task['appear_time'], TASK_ARRIVAL, payload)

    def handle(self, event_id: Any, payload: Any):
        if event_id == TASK_ARRIVAL:
            self.handle_task_arrival(payload)

        elif event_id == DASHER_ARRIVAL:
            self.handle_dasher_arrival(payload)

        elif event_id == SIMULATION_END:
            self.handle_simulation_end(payload)

    # probably less relevant because "requests" go in the opposite direction now
    # -> yes this code ended up being much shorter (car request got repurposed into task arrival)
    def handle_task_arrival(self, payload: Any):
        task = payload['task']
        self.available_tasks.append(task)

    # opportunity cost algorithm, differs from baseline here
    def handle_dasher_arrival(self, payload: Any):
        dasher_id = payload['dasher_id']
        location = payload['location']

        # if the dasher has arrived, complete task
        ## used copilot to refine the code here, because I kept on running into the issue where the simulation wouldn't correctly mark tasks as 'completed' and free up the dashers.
        current_goal = self.dashers[dasher_id]['current_goal']
        if current_goal is not None and location == current_goal['location']:
            self.total_score += current_goal['reward']          
            if current_goal in self.available_tasks:
                self.available_tasks.remove(current_goal)          
            self.dashers[dasher_id]['current_goal'] = None      
            self.dashers[dasher_id]['goal_expiration'] = None   

        # update dasher location and status
        self.dashers[dasher_id]['location'] = location
        self.dashers[dasher_id]['status'] = 'available'

        # check if dasher has exited yet
        if self.now >= self.dashers[dasher_id]['exit_time']:
            self.dashers_exited += 1
            if self.dashers_exited == self.num_dashers:
                self.schedule_at(self.now, SIMULATION_END, None)
            return

        # 1- filter tasks that dasher can reach in time
        feasible_tasks = []
        for task in self.available_tasks:
            path, travel_time = self.graph.dijkstra_shortest_path(location, task['location'])
            if path:
                arrival_time = self.now + travel_time
                if arrival_time <= task['target_time'] and arrival_time <= self.dashers[dasher_id]['exit_time']:
                    feasible_tasks.append((task, travel_time))  # CHANGED: removed arrival_time

        if not feasible_tasks:
            return  # no feasible tasks available
        
        # 2- prioritize feasible task with highest reward
        feasible_tasks.sort(key=lambda x: x[0]['reward'], reverse=True)
        best_task = None #must initialize before entering the loop
        best_travel = None #same here

        # 3- assign task if not already covered
        # this prevents overcrowding at tasks, making sure that dashers are at least headed toward different tasks
        for task, travel_time in feasible_tasks:
            covered = False
            for d in self.dashers.values():
                if d['current_goal'] == task:
                    covered = True
                    break
            if not covered:
                best_task = task
                best_travel = travel_time
                break

        # 4- if task is covered, fall back to closest feasible task
        ## this fallback mechanism is not ideal, because it just immediately goes back to the closest task. But at the same time I think this allows the dashers to be freed up quickly for the next possible high-reward task
        if best_task is None:
            best_task, best_travel = min(feasible_tasks, key=lambda x: x[1])

        # 5- assign task to dasher
        self.dashers[dasher_id]['status'] = 'unavailable'
        self.dashers[dasher_id]['current_goal'] = best_task
        self.dashers[dasher_id]['goal_expiration'] = best_task['target_time']

        self.schedule_at(
            self.now + best_travel,
            DASHER_ARRIVAL,
            {
                'dasher_id': dasher_id,
                'location': best_task['location']
            }
        )


    # actually didn't need a check_for_simulation_end since it has a set end
    # handle_simulation_end can be exactly the same
    def handle_simulation_end(self, payload: Any):
        """
        Stops the simulation.
        """
        print("SIMULATION_END event received. Stopping run loop.")
        self.stop()

    def get_total_score(self):
        """ return the total system score """
        return self.total_score

class BatchingSimulator(Simulator):
    """Implements the Batching strategy as described in the submitted Smart Brain paper."""
    def __init__(self, graph: WeightedGraph, dashers, tasks, batch_interval: int = 5):
        super().__init__()
        self.graph = graph
        self.dashers_data = dashers
        self.tasks_data = tasks
        self.batch_interval = batch_interval
        self.available_tasks = []
        self.dashers = {}
        self.total_score = 0.0
        self.num_dashers = len(dashers)
        self.dashers_exited = 0
        self.pending_dashers = []
    
    def start_simulation(self):
        for i, dasher_info in enumerate(self.dashers_data):
            self.dashers[i] = {
                'location': dasher_info['start_location'],
                'exit_time': dasher_info['exit_time'],
                'status': 'available',
                'current_goal': None
            }
        payload = {'dasher_id': i, 'location': dasher_info['start_location']}
        self.schedule_at(dasher_info['start_time'], DASHER_ARRIVAL, payload)

        for task in self.tasks_data:
            self.schedule_at(task['appear_time'], TASK_ARRIVAL, {'task': task})
        
        earliest = min(d['start_time'] for d in self.dashers_data)
        latest = max(d['exit_time'] for d in self.dashers_data)
        t = earliest
        while t <= latest:
            self.schedule_at(t, BATCH_DISPATCH, None)
            t += self.batch_interval

    def handle(self, event_id, payload):
        if event_id == TASK_ARRIVAL:
            self.available_tasks.append(payload['task'])
        elif event_id == DASHER_ARRIVAL:
            self.handle_dasher_arrival(payload)
        elif event_id == BATCH_DISPATCH:
            self.handle_batch_dispatch()
        elif event_id == SIMULATION_END:
            print("SIMULATION_END event received. Stopping run loop.")
            self.stop()
    
    def handle_dasher_arrival(self, payload):
        dasher_id = payload['dasher_id']
        location = payload['location']
        current_goal = self.dashers[dasher_id]['current_goal']
        if current_goal is not None and location == current_goal['location']:
            self.total_score += current_goal['reward']
            if current_goal in self.available_tasks:
                self.available_tasks.remove(current_goal)
            self.dashers[dasher_id]['current_goal'] = None
        self.dashers[dasher_id]['location'] = location
        self.dashers[dasher_id]['status'] = 'available'
        if self.now >= self.dashers[dasher_id]['exit_time']:
            self.dashers_exited += 1
            if self.dashers_exited == self.num_dashers:
                self.schedule_at(self.now, SIMULATION_END, None)
            return
        self.pending_dashers.append(dasher_id)
    
    def handle_batch_dispatch(self):
        """Dispatch all pending dashers."""
        for dasher_id in self.pending_dashers:
            if self.dashers[dasher_id]['status'] != 'available':
                continue
            if self.now >= self.dashers[dasher_id]['exit_time']:
                continue
            
            location = self.dashers[dasher_id]['location']

            feasible_tasks = []
            for task in self.available_tasks:
                path, travel_time = self.graph.dijkstra_shortest_path(location, task['location'])
                if path:
                    arrival_time = self.now + travel_time
                    if arrival_time <= task['target_time'] and arrival_time <= self.dashers[dasher_id]['exit_time']:
                        feasible_tasks.append((task, travel_time))
            
            if not feasible_tasks:
                continue
            
            feasible_tasks.sort(key=lambda x: x[0]['reward'], reverse=True)

            best_task, best_travel = None, None
            for task, travel_time in feasible_tasks:
                covered = any(d['current_goal'] == task for d in self.dashers.values())
                if not covered:
                    best_task, best_travel = task, travel_time
                    break
            if best_task is None:
                best_task, best_travel = min(feasible_tasks, key=lambda x: x[1])

            self.dashers[dasher_id]['status'] = 'unavailable'
            self.dashers[dasher_id]['current_goal'] = best_task
            self.schedule_at(self.now + best_travel, DASHER_ARRIVAL,
                {'dasher_id': dasher_id, 'location': best_task['location']})
        self.pending_dashers = []
    
    def get_total_score(self):
        return self.total_score

if __name__ == "__main__":
####used Gemini for this because I didn't really know how to write a main execution block
## ^ this note was for the original version: new version was handwritten since I didn't
## need to learn to use sys input or anything
    graph_file = "project_files/grid100.txt"
    dashers_file = "project_files/dashers.csv"
    tasklog_file = "project_files/tasklog.csv"
    # from the assignment doc... "we can choose to keep the 'appear time' fixed
    # (e.g. appears 20min before its reported completion)". So I added this feature?
    appear_time_fixed = 20

    graph = read_graph(graph_file)

    dashers = read_dashers(dashers_file)

    tasks = read_tasks(tasklog_file, appear_time_fixed)

    sim_base = Baseline(graph, dashers, tasks)
    sim_base.start_simulation()
    sim_base.run()
    base_total_score = sim_base.get_total_score()
    print(f"\nBaseline Total Score: {base_total_score:.2f}")

    
    sim_opp = OpportunityCost(graph, dashers, tasks)
    sim_opp.start_simulation()
    sim_opp.run()
    opp_total_score = sim_opp.get_total_score()
    print(f"\nOpportunity Cost Total Score: {opp_total_score:.2f}")