from docsaidkit import COLORSTR, FORMATSTR, colorstr, make_batch


def number_generator(start, end):
    for i in range(start, end + 1):
        yield i

def test_make_batch():
    # Create a number generator from 1 to 10
    data_generator = number_generator(1, 10)

    # Batch size of 3
    batch_size = 3

    # Generate batched data
    batched_data_generator = make_batch(data_generator, batch_size)

    # Check the first batch
    batch = next(batched_data_generator)
    assert batch == [1, 2, 3]

    # Check the second batch
    batch = next(batched_data_generator)
    assert batch == [4, 5, 6]

    # Check the third batch
    batch = next(batched_data_generator)
    assert batch == [7, 8, 9]

    # Check the fourth batch (last batch with remaining data)
    batch = next(batched_data_generator)
    assert batch == [10]

def test_colorstr_blue_bold():
    obj = "Hello, colorful world!"
    expected_output = "\033[1;34mHello, colorful world!\033[0m"
    assert colorstr(obj, color=COLORSTR.BLUE, fmt=FORMATSTR.BOLD) == expected_output

def test_colorstr_red():
    obj = "Error: Something went wrong!"
    expected_output = "\033[1;31mError: Something went wrong!\033[0m"
    assert colorstr(obj, color='red') == expected_output

def test_colorstr_underline_green():
    obj = "Important message"
    expected_output = "\033[4;32mImportant message\033[0m"
    assert colorstr(obj, color=32, fmt='underline') == expected_output

def test_colorstr_default():
    obj = "Just a regular text"
    expected_output = "\033[1;34mJust a regular text\033[0m"
    assert colorstr(obj) == expected_output
