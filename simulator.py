from typing import Any, Optional, Tuple
from collections import defaultdict
import heapq
import itertools
import random

CAR_REQUEST = "CAR_REQUEST"
CAR_ARRIVAL = "CAR_ARRIVAL"
SIMULATION_END = "SIMULATION_END"

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
            raise Exception("Not a valid edge")

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

        graph.addEdge(node1, node2, weight)
    
    # Close the file safely after done reading
    file.close()
    return graph

def read_agents(fname):
    # Open the file
    file = open(fname, "r")
    # Set up your list of agents
    agents=[]

    # Next, read the agents and build the list
    for line in file:
        # agent is a list of 2 indices representing a pair of vertices
        # path[0] contains the start location (index between 0 and numVertices-1)
        # path[1] contains the destination location (index between 0 and numVertices-1)
        path = line.strip().split(",")
        agents.append((int(path[0]), int(path[1])))
    
    # Close the file safely after done reading
    file.close()
    return agents

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
# Generates a random integer N such that 1 <= N <= 10
    def __init__(self, graph: WeightedGraph, car_requests: list):
        super().__init__()
        self.graph = graph
        self.car_requests = car_requests
        self.edge_occ = defaultdict(int)
        self.cars = {}
        self.total_system_congestion = 0.0
        self.car_id_count = 0
        self.num_cars = len(self.car_requests)
        self.cars_finished_count = 0


    def start_simulation(self):
        """Schedules the initial CAR_REQUEST events."""
        current_t=0.0
        for (start_node, end_node) in self.car_requests:
            #If its not the first car, generate a random time
            if self.car_id_count > 0:
                inter_arrival_time = random.expovariate(1.0)
                current_t += inter_arrival_time

            #assign its id
            self.car_id_count += 1
            car_id = self.car_id_count
            
            #payload encodes car info, ex. 'C1' in class
            payload = {
                'car_id': car_id,
                'start_node': start_node,
                'end_node': end_node
            }
            # Schedule the car's arrival in the system
            self.schedule_at(current_t, CAR_REQUEST, payload)
            
            #initialize car data for stats
            self.cars[car_id] = {
                'start': start_node,
                'end': end_node,
                'arrival_time': current_t,
                'path': None,
                'path_record': [],
                'final_arrival_time': None #adding this just to track it
            }

    def handle(self, event_id: Any, payload: Any):

        if event_id == CAR_REQUEST:
            self.handle_car_request(payload)
            
        elif event_id == CAR_ARRIVAL:
            self.handle_car_arrival(payload)
            
        elif event_id == SIMULATION_END: 
            self.handle_simulation_end(payload)

    # probably less relevant because "requests" go in the opposite direction now
    def handle_car_request(self, payload: Any):
        car_id = payload['car_id']
        start = payload['start_node']
        end = payload['end_node']

        car_path, base = self.graph.dijkstra_shortest_path(start, end)
    
        if not car_path:
            print(f"No path found from {start} to {end} for Car {car_id}. Car removed.")
            self.check_for_simulation_end()
            return

        self.cars[car_id]['path'] = car_path
    
        # Put car at start node 
        # Activate the traversal
        start_payload = {
            'car_id': car_id,
            'at_node': start,
            'prev_node': None # Just started, no previous edge
        }
        self.schedule_at(self.now, CAR_ARRIVAL, start_payload)
            

    # can probably reuse to an extent
    def handle_car_arrival(self, payload: Any):
        car_id = payload['car_id']
        curr_node = payload['at_node']
        prev_node = payload['prev_node']
        car_data = self.cars[car_id]
        
        if prev_node is not None:
            finished_edge = (prev_node, curr_node)
            self.edge_occ[finished_edge] -= 1

        if curr_node == car_data['end']:
            print(f"[{self.now:.2f}] Car {car_id} reached destination {curr_node}.")
            # need to record the finish time before returning if we add tracking the final arrival time
            car_data['final_arrival_time'] = self.now
            self.check_for_simulation_end()
            return
            
        predetermined_path = car_data['path']
        try:
            curr_node_index = predetermined_path.index(curr_node)
            next_node = predetermined_path[curr_node_index + 1]
        except (ValueError, IndexError):
            print(f" Car {car_id} at {curr_node}, path {predetermined_path} error.")
            self.check_for_simulation_end()
            return
            
        next_edge = (curr_node, next_node)
        base_weight = self.graph.get_weight(curr_node, next_node)
        
        if base_weight is None:
            print(f"Error: Edge ({curr_node}, {next_node}) not found")
            return
        
        k = self.edge_occ[next_edge] + 1
        self.edge_occ[next_edge] = k
        
        traversal_time = base_weight * k
        
        # Cost for this car is the time it took
        car_data['path_record'].append((next_edge, traversal_time))
        
        # System congestion cost (marginal cost): w*(k^2) - w*((k-1)^2) = w*(2k - 1)
        marginal_system_cost = base_weight * (2 * k - 1)
        self.total_system_congestion += marginal_system_cost
        
        print(f"[{self.now:.2f}] Car {car_id} starting edge {next_edge}. (k={k}, T={traversal_time:.2f})")

        # --- 6. Schedule arrival at the *next* node ---
        arrival_time = self.now + traversal_time
        next_payload = {
            'car_id': car_id,
            'at_node': next_node,
            'prev_node': curr_node # This will be the edge to decrement
        }
        self.schedule_at(arrival_time, CAR_ARRIVAL, next_payload)

    # can probably be reused (no more availsble dashers)
    def check_for_simulation_end(self):
        self.cars_finished_count += 1
        if self.cars_finished_count == self.num_cars:
            print(f"[{self.now:.2f}] All {self.num_cars} cars have finished. Scheduling end.")
            self.schedule_at(self.now, SIMULATION_END, None)

    # can be exactly the same
    def handle_simulation_end(self, payload: Any):
        """
        Stops the simulation.
        """
        print(f"[{self.now:.2f}] SIMULATION_END event received. Stopping run loop.")
        self.stop()

    def print_statistics(self):
        total_car_costs = 0.0
        num_cars = 0
        
        for car_id, car in self.cars.items():
            if not car['path_record']: 
                continue
                
            num_cars += 1
            path_str_list = []
            car_total_cost = 0.0
            
            for (edge, time_taken) in car['path_record']:
                path_str_list.append(f"({edge[0]}-{edge[1]},{time_taken:.2f})")
                car_total_cost += time_taken
            
            path_str = ", ".join(path_str_list)
            final_time = car.get('final_arrival_time', car['arrival_time'])
            print(f"Car {car_id} ({car['start']}, {car['end']}), arrived at t={final_time:.2f}, with path {path_str}")
            total_car_costs += car_total_cost
            
        avg_congestion = total_car_costs / num_cars 
        
        print(f"\nAverage congestion is {avg_congestion:.2f}")
        print(f"Total congestion is {self.total_system_congestion:.2f}")
        
        return avg_congestion, self.total_system_congestion

class Dasher_Baseline(Simulator):
# Generates a random integer N such that 1 <= N <= 10
    def __init__(self, graph: WeightedGraph, dashers: list,tasks: list):
        super().__init__()
        self.graph = graph
        self.dashers = dashers
        self.edge_occ = defaultdict(int)
        self.cars = {}
        self.total_system_congestion = 0.0
        self.car_id_count = 0
        self.num_cars = len(self.car_requests)
        self.cars_finished_count = 0

    def get_task_data(self, tasklogfile):
        """ get tasklogdata and saves into csv taskfile that includes task data we need"""
    
        file = open(tasklogfile, "r")
        # Set up your list of agents
        tasks=[]

        # Next, read the agents and build the list
        for line in file:
            # agent is a list of 2 indices representing a pair of vertices
            # path[0] contains the start location (index between 0 and numVertices-1)
            # path[1] contains the destination location (index between 0 and numVertices-1)
            path = line.strip().split(",")
#             USERID,VERTEX,TIME,minute
# 470,38,2012-04-03 18:00:09+00:00,1080
            tasks.append((int(path[0]), int(path[1]),int(path[3])))
            # userid, location(vertex),target_time
        
        # Close the file safely after done reading
        file.close()
        return agents

    def start_simulation(self):
        """Schedules the initial CAR_REQUEST events."""
        current_t=0.0
        for (start_node, end_node) in self.car_requests:
            #If its not the first car, generate a random time
            if self.car_id_count > 0:
                inter_arrival_time = random.expovariate(1.0)
                current_t += inter_arrival_time

            #assign its id
            self.car_id_count += 1
            car_id = self.car_id_count
            
            #payload encodes car info, ex. 'C1' in class
            payload = {
                'car_id': car_id,
                'start_node': start_node,
                'end_node': end_node
            }
            # Schedule the car's arrival in the system
            self.schedule_at(current_t, CAR_REQUEST, payload)
            
            #initialize car data for stats
            self.cars[car_id] = {
                'start': start_node,
                'end': end_node,
                'arrival_time': current_t,
                'path': None,
                'path_record': [],
                'final_arrival_time': None #adding this just to track it
            }

    def handle(self, event_id: Any, payload: Any):

        if event_id == TASK_ARRIVAL:
            self.handle_task_arrival(payload)
            
        elif event_id == DASHER_ARRIVAL:
            self.handle_dasher_arrival(payload)
            
        elif event_id == SIMULATION_END: 
            self.handle_simulation_end(payload)

    # probably less relevant because "requests" go in the opposite direction now
    def handle_car_request(self, payload: Any):
        car_id = payload['car_id']
        start = payload['start_node']
        end = payload['end_node']

        car_path, base = self.graph.dijkstra_shortest_path(start, end)
    
        if not car_path:
            print(f"No path found from {start} to {end} for Car {car_id}. Car removed.")
            self.check_for_simulation_end()
            return

        self.cars[car_id]['path'] = car_path
    
        # Put car at start node 
        # Activate the traversal
        start_payload = {
            'car_id': car_id,
            'at_node': start,
            'prev_node': None # Just started, no previous edge
        }
        self.schedule_at(self.now, CAR_ARRIVAL, start_payload)
            

    # can probably reuse to an extent
    def handle_car_arrival(self, payload: Any):
        car_id = payload['car_id']
        curr_node = payload['at_node']
        prev_node = payload['prev_node']
        car_data = self.cars[car_id]
        
        if prev_node is not None:
            finished_edge = (prev_node, curr_node)
            self.edge_occ[finished_edge] -= 1

        if curr_node == car_data['end']:
            print(f"[{self.now:.2f}] Car {car_id} reached destination {curr_node}.")
            # need to record the finish time before returning if we add tracking the final arrival time
            car_data['final_arrival_time'] = self.now
            self.check_for_simulation_end()
            return
            
        predetermined_path = car_data['path']
        try:
            curr_node_index = predetermined_path.index(curr_node)
            next_node = predetermined_path[curr_node_index + 1]
        except (ValueError, IndexError):
            print(f" Car {car_id} at {curr_node}, path {predetermined_path} error.")
            self.check_for_simulation_end()
            return
            
        next_edge = (curr_node, next_node)
        base_weight = self.graph.get_weight(curr_node, next_node)
        
        if base_weight is None:
            print(f"Error: Edge ({curr_node}, {next_node}) not found")
            return
        
        k = self.edge_occ[next_edge] + 1
        self.edge_occ[next_edge] = k
        
        traversal_time = base_weight * k
        
        # Cost for this car is the time it took
        car_data['path_record'].append((next_edge, traversal_time))
        
        # System congestion cost (marginal cost): w*(k^2) - w*((k-1)^2) = w*(2k - 1)
        marginal_system_cost = base_weight * (2 * k - 1)
        self.total_system_congestion += marginal_system_cost
        
        print(f"[{self.now:.2f}] Car {car_id} starting edge {next_edge}. (k={k}, T={traversal_time:.2f})")

        # --- 6. Schedule arrival at the *next* node ---
        arrival_time = self.now + traversal_time
        next_payload = {
            'car_id': car_id,
            'at_node': next_node,
            'prev_node': curr_node # This will be the edge to decrement
        }
        self.schedule_at(arrival_time, CAR_ARRIVAL, next_payload)

    # can probably be reused (no more availsble dashers)
    def check_for_simulation_end(self):
        self.cars_finished_count += 1
        if self.cars_finished_count == self.num_cars:
            print(f"[{self.now:.2f}] All {self.num_cars} cars have finished. Scheduling end.")
            self.schedule_at(self.now, SIMULATION_END, None)

    # can be exactly the same
    def handle_simulation_end(self, payload: Any):
        """
        Stops the simulation.
        """
        print(f"[{self.now:.2f}] SIMULATION_END event received. Stopping run loop.")
        self.stop()

    def print_statistics(self):
        total_car_costs = 0.0
        num_cars = 0
        
        for car_id, car in self.cars.items():
            if not car['path_record']: 
                continue
                
            num_cars += 1
            path_str_list = []
            car_total_cost = 0.0
            
            for (edge, time_taken) in car['path_record']:
                path_str_list.append(f"({edge[0]}-{edge[1]},{time_taken:.2f})")
                car_total_cost += time_taken
            
            path_str = ", ".join(path_str_list)
            final_time = car.get('final_arrival_time', car['arrival_time'])
            print(f"Car {car_id} ({car['start']}, {car['end']}), arrived at t={final_time:.2f}, with path {path_str}")
            total_car_costs += car_total_cost
            
        avg_congestion = total_car_costs / num_cars 
        
        print(f"\nAverage congestion is {avg_congestion:.2f}")
        print(f"Total congestion is {self.total_system_congestion:.2f}")
        
        return avg_congestion, self.total_system_congestion

if __name__ == "__main__":
####used Gemini for this because I didn't really know how to write a main execution block
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python Simulator_Fixed.py <graph_file> <agents_file>")
        sys.exit(1)

    graph_file = sys.argv[1]
    agents_file = sys.argv[2]

    print(f"Loading graph from {graph_file}...")
    graph = read_graph(graph_file)
    print(f"  Loaded {len(graph.getNodes())} nodes")

    print(f"Loading agents from {agents_file}...")
    agents = read_agents(agents_file)
    print(f"  Loaded {len(agents)} car requests")

    print("\nStarting simulation...")

    sim = Baseline(graph, agents)
    sim.start_simulation()
    sim.run()

    print("\nSimulation Results:")
    sim.print_statistics()