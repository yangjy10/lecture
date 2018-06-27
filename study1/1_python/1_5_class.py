'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

class operation:
    def __init__(self):
        self.result = 0
    def add(self, a, b):
        self.result = a+b
    def div(self, a, b):
        self.result = a/b
    def get(self):
        return self.result

op = operation()
op.add(1,2)
print(op.get())
