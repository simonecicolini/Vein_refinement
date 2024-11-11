
class cell:
    def __init__(self, position, u, index):

        self.position = position
        self.u =u
        self.index = index
        self.neighbours = []
        self.second_neighbours = []
        self.color = 'blue'
