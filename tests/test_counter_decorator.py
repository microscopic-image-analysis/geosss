from geosss.utils import count_calls, counter


@counter(["my_method", "another_method"])
class MyClass:
    def my_method(self):
        pass

    def another_method(self):
        pass


@counter("my_method")
class MySubClass(MyClass):
    def my_method(self):
        pass


@count_calls
def test_func():
    pass


if __name__ == '__main__':

    a = MyClass()
    [a.my_method() for _ in range(2)]
    a.another_method()

    print(f" In total, {a.my_method.__name__} was called "
          f"{a.my_method.num_calls} times")
    print(f" In total, {a.another_method.__name__} was called "
          f"{a.another_method.num_calls} times")

    b = MySubClass()
    [b.my_method() for _ in range(2)]
    print(f" In total, {b.my_method.__name__} from subclass was called "
          f"{b.my_method.num_calls} times")

    [test_func() for _ in range(3)]
    print(f" In total, function {test_func.__name__} was called "
          f"{test_func.num_calls} times")
