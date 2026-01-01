import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_dist, current_node = heapq.heappop(queue)
        if current_dist > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return distances

# 그래프 정의
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2},
    'C': {'A': 4, 'B': 2}
}

print(dijkstra(graph, 'A'))
