'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

# package
import package.ex
print(package.ex.example())

from package import ex
print(ex.example())

from package import * # package.__init__.py
print(ex.example())


# sub package
import package.sub.subex
print(package.sub.subex.example())

from package.sub import subex
print(subex.example())

from package.sub.subex import example
print(example())

from package.sub import * # package.sub.__init__.py
print(subex.example())
