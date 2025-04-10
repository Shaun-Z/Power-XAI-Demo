import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

ring = False # if ring structure

def generate_incidence_matrix(n_buses, prob=0.2):
    """Generates an incidence matrix for a ring structure"""
    branches = []
    for i in range(n_buses-1):
        branches.append((i, i+1))  # Ring connection
    
    # Step 2: Add extra random connections
    if not ring:
        all_possible_connections = [(i, j) for i in range(1,n_buses) for j in range(i + 2, n_buses)]
        for (i, j) in all_possible_connections:
            if np.random.rand() > prob:
                continue
            if (i, j) not in branches and (j, i) not in branches:  # Avoid duplicates
                branches.append((i, j))
    
    # create a loop
    branches.append((n_buses-1, 0))

    A = np.zeros((len(branches), n_buses))
    for idx, (i, j) in enumerate(branches):
        A[idx, i] = 1
        A[idx, j] = -1  # Directed branch
    
    return A, branches

def compute_Y(A, Y_e, slack_bus=0):
    """Computes the bus admittance matrix Y and removes the slack bus row/column"""
    # Compute bus admittance matrix Y = A^T Y_e A
    Y = A.T @ Y_e @ A

    # Remove slack bus row and column (Bus 1 → index 0)
    Y_reduced = np.delete(Y, slack_bus, axis=0)  # Remove row
    Y_reduced = np.delete(Y_reduced, slack_bus, axis=1)  # Remove column

    return Y_reduced

def generate_voltage_data(n_buses, I, A=None, Y_e=None, outage=False):
    """Generates voltage data using V = I Y^{-1} with Bus 1 as the reference"""
    
    if not outage:
        print("[Normal Data]")
        # Generate the incidence matrix
        A, branches = generate_incidence_matrix(n_buses)

        # Generate random admittance values for each branch
        branch_admittances = np.random.uniform(0.4, 0.6, size=len(branches))  # Random admittances
        Y_e = np.diag(branch_admittances)  # Admittance diagonal matrix

    # Compute the reduced admittance matrix after removing the slack bus
    Y_reduced = compute_Y(A, Y_e, slack_bus=0)

    # Compute the inverse admittance matrix (Y is now invertible)
    Y_inv = np.linalg.inv(Y_reduced)

    # Compute voltages using V = I Y^{-1} (excluding reference bus)
    V = I @ Y_inv  # (time_steps, n_buses-1)

    # # Add small noise
    # V += np.random.normal(0, 0.01, size=V.shape)

    print("Incidence Matrix A:\n", A)
    print("Admittance Matrix Y:\n", Y_reduced)
    print("Y_inv:\n", Y_inv)
    print("Voltage Mean (Estimated):\n", np.mean(V, axis=0))
    print("Voltage Covariance:\n", np.cov(V, rowvar=False))

    return V.T, Y_reduced, A, Y_e  # Return voltages, admittance matrix, incidence matrix, and branch admittance

def outage(A, Y_e):
    """Simulates a line outage by removing a random branch"""
    branch_idx = np.random.choice(np.arange(1, A.shape[0] - 1))  # Randomly pick a branch
    
    from_bus = np.where(A[branch_idx] == 1)[0][0]+1  # Bus where the branch starts
    to_bus = np.where(A[branch_idx] == -1)[0][0]+1  # Bus where the branch ends
    print(f"\n[Outage Occurred: Removing Branch between (bus {from_bus} and bus {to_bus})]")

    # Remove the selected branch from A and Y_e
    A = np.delete(A, branch_idx, axis=0)
    Y_e = np.delete(Y_e, branch_idx, axis=0)
    Y_e = np.delete(Y_e, branch_idx, axis=1)

    # """Simulates another line outage by removing a random branch"""
    # branch_idx_2 = np.random.choice(np.arange(1, A.shape[0] - 1))  # Randomly pick another branch
    # from_bus = np.where(A[branch_idx_2] == 1)[0][0]+1  # Bus where the branch starts
    # to_bus = np.where(A[branch_idx_2] == -1)[0][0]+1  # Bus where the branch ends
    # print(f"\n[Outage Occurred: Removing Branch between (bus {from_bus} and bus {to_bus})]")
    # # Remove the selected branch from A and Y_e
    # A = np.delete(A, branch_idx_2, axis=0)
    # Y_e = np.delete(Y_e, branch_idx_2, axis=0)
    # Y_e = np.delete(Y_e, branch_idx_2, axis=1)

    return A, Y_e

def draw_grid_from_incidence(A):
    num_branches, num_buses = A.shape
    G = nx.Graph()

    # Add buses (nodes)
    for bus in range(num_buses):
        G.add_node(bus + 1)  # Nodes indexed from 1

    # Add branches (edges)
    for branch_idx in range(num_branches):
        from_bus = np.where(A[branch_idx] == 1)[0][0]  # Bus where branch starts
        to_bus = np.where(A[branch_idx] == -1)[0][0]  # Bus where branch ends
        G.add_edge(from_bus + 1, to_bus + 1)  # Convert to 1-based indexing

    # Draw the graph
    pos = nx.spring_layout(G)  # Generate layout
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)


if __name__ == "__main__":
    import os
    import matplotlib

    os.makedirs('data', exist_ok=True)
    matplotlib.use('Agg')  # Use non-interactive backend for saving figures

    # Parameters
    n_buses = 10  # Number of buses
    time_steps = 1000  # Number of time steps

    # Generate independent Gaussian current injections (excluding the slack bus)
    I = np.random.normal(0, 0.2, size=(time_steps, n_buses - 1))  # (time_steps, n_buses-1)

    # Generate normal voltage data
    voltage_data, Y_reduced, A, Y_e = generate_voltage_data(n_buses, I)
    np.save('./data/Y_matrix.npy', Y_reduced)

    # Simulate outage and generate new voltage data
    A_outage, Y_e_outage = outage(A.copy(), Y_e.copy())
    voltage_data_outage, Y_outage, A_outage, Y_e_outage = generate_voltage_data(n_buses, I, A=A_outage, Y_e=Y_e_outage, outage=True)

    # Save normal voltage data to CSV
    voltage_df = pd.DataFrame(voltage_data.T, columns=[f'Bus_{i+2}' for i in range(n_buses-1)])  # Bus 1 is removed
    voltage_df.to_csv('./data/voltage_data.csv', index=False)

    # Save outage voltage data to CSV
    voltage_outage_df = pd.DataFrame(voltage_data_outage.T, columns=[f'Bus_{i+2}' for i in range(n_buses-1)])  # Bus 1 is removed
    voltage_outage_df.to_csv('./data/voltage_data_outage.csv', index=False)


    # Plot the adjacency matrices
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.title("Original Adjacency Matrix")
    draw_grid_from_incidence(A)
    plt.xlabel("Bus Index")
    plt.ylabel("Bus Index")

    plt.subplot(122)
    plt.title("Adjacency Matrix After Outage")
    draw_grid_from_incidence(A_outage)
    plt.xlabel("Bus Index")
    plt.ylabel("Bus Index")

    plt.tight_layout()
    plt.savefig("./data/adjacency_matrices.png")    
    plt.close()
    

    # Plot the voltage time series for a few buses
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    for i in range(min(n_buses-1, 50)):  # Plot first 5 buses
        plt.plot(voltage_data[i, :], label=f'Bus {i+2}')
    plt.xlabel("Time Steps")
    plt.ylabel("Voltage Magnitude")
    plt.title("Simulated Voltage Data with Different Means")
    plt.legend()

    plt.subplot(122)
    for i in range(min(n_buses-1, 50)):  # Plot first 5 buses
        plt.plot(voltage_data_outage[i, :], label=f'Bus {i+2}')
    plt.xlabel("Time Steps")
    plt.ylabel("Voltage Magnitude")
    plt.title("Simulated Voltage Data with Different Means")
    plt.legend()

    plt.savefig("./data/voltage_data.png")
    plt.close()
    print("Data generation completed.")