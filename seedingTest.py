class Points:
    a = []
    def __init__(self, scope):
        self.scope = scope
    def generate(self):
        return self.scope

p = Points(20)
print(p.generate())
