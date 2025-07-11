from functools import cached_property


class Foo:
    def __init__(self):
        self.a = 1
        self.b = 2

    @cached_property
    def c(self):
        return self.a + self.b

    def __setattr__(self, name, value):
        if name == "c":
            raise AttributeError("Can't set read-only property 'c'")
        super().__setattr__(name, value)


foo1 = Foo()
print(foo1.a, foo1.b, foo1.c)  # 1,2, 12
# foo1.c = 12  # this should be readonly propery
print(foo1.a, foo1.b, foo1.c)  # 1,2, 12 BUG!!
