from contextlib import redirect_stdout
import io


def catch_print(func, **kwargs):
    output_capture = io.StringIO()
    with redirect_stdout(output_capture):
        result = func(**kwargs)
    output = output_capture.getvalue()
    output_capture.close()
    return output,result

if __name__ == '__main__':
    def greet(name):
        print(f"Hello, {name}!")
        return 42

    # 使用 catch_print 函数
    output, result = catch_print(greet, name="Alice")

    print("Captured Output:", output)  # 输出: Captured Output: Hello, Alice!
    print("Function Result:", result)  # 输出: Function Result: 42