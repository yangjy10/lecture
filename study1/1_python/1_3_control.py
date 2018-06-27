'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

condition = 1

print("if 문")
if condition == 1:
    print("1")
elif condition == 2:
    print("2")
else:
    print("else")

print("while 문")
# while 문
while condition <= 10:
    print(condition)
    condition = condition + 1

print("for 문")
conditions = [1,2,3]
for i in conditions:
    print(i)

