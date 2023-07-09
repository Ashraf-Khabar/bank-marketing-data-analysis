import time

def detect_sinkhole(network):
    # Iterate over each node in the network
    for node in network:
        # Initialize an empty list to store jump information
        jumps = []
        
        # Initialize an abnormal variation threshold
        threshold = 5  # Adjust the threshold value as needed
        
        # Continue loop as long as the node is active
        while node.is_active:
            # If the node receives a data packet towards a given destination
            if node.receives_packet:
                destination = node.get_destination()
                
                # If the destination is not already present in jumps
                if destination not in jumps:
                    # Add the destination to jumps with the observed jump count
                    jumps.append((destination, 1))
                else:
                    # Retrieve the previously recorded jump count for the destination
                    previous_jumps = None
                    for jump in jumps:
                        if jump[0] == destination:
                            previous_jumps = jump[1]
                            break
                    
                    # Compare the current jump count with the previous count
                    current_jumps = node.get_jump_count(destination)
                    difference = current_jumps - previous_jumps
                    
                    # If the difference exceeds the threshold
                    if difference > threshold:
                        # Report an anomaly detection for the node and destination
                        report_anomaly(node, destination)
                    
                    # Update the jump count in jumps for the destination
                    for i, jump in enumerate(jumps):
                        if jump[0] == destination:
                            jumps[i] = (destination, current_jumps)
                            break
            
            # Wait for a certain time before repeating the process
            wait(1)  # Adjust the wait time as needed

# Define a Node class to represent network nodes
class Node:
    def __init__(self):
        self.is_active = True
        self.receives_packet = False
        self.destination = None
        self.jump_counts = {}
        
    def get_destination(self):
        return self.destination
    
    def get_jump_count(self, destination):
        return self.jump_counts.get(destination, 0)
    

# Function to report anomaly detection
def report_anomaly(node, destination):
    print(f"Anomaly detected: Node {node} to Destination {destination}")


# Function to simulate waiting
def wait(seconds):
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= seconds:
            break
        try:
            time.sleep(0.1)  # Sleep in smaller increments to allow for interrupts
        except KeyboardInterrupt:
            break


# Function to create a sample network
def create_network():
    network = []
    node1 = Node()
    node2 = Node()
    node3 = Node()
    
    # Configure node properties
    node1.destination = "A"
    node1.jump_counts = {"A": 1}
    
    node2.destination = "B"
    node2.jump_counts = {"B": 2}
    
    node3.destination = "C"
    node3.jump_counts = {"C": 3}
    
    network.extend([node1, node2, node3])
    return network


def main():
    # Create a sample network
    network = create_network()
    
    # Call the detection function
    detect_sinkhole(network)


# Call the main function
main()

