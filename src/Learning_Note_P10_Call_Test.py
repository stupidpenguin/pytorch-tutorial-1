# 解釋 call 的使用方法
class Person:
    def __call__(self, name):
        print("__call__"+" Hello " + name)

    def hello(self, name):
        print("hello" + name)


person = Person()
person("Tom")
person.hello("lisi")