'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

# 튜플
tuple = (1, 2, '1', '2')
print(tuple[0]) # 숫자 출력 : type(tuple[0])
print(tuple[3]) # 문자 출력 : type(tuple[0])

print(tuple + (3, '3'))
print(tuple[:2])

#del(tuple[3]) # 에러 : 튜플은 수정불가
#tuple.append(3) # 에러 : 튜플은 수정불가

tuple1 = tuple[0:2]
print(tuple1)


# 리스트
list = [1, 2, '1', '2']
print(list)

#print(list + (3, '3')) # 에러 같은 형끼리만 연산이 된다
print(list + [3, '3']) 

del(list[3])
list.append(3) 
print(list)

# 사전형
dic = {'1':2, 1:'2'}
print(dic['1'])

dic = {'t': [1,2,3]}
print(dic['t'][1])
