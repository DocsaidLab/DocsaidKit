import pytest

from docsaidkit import PowerDict

power_dict_attr_data = [
    (
        {
            'key': 'value',
            'name': 'mock_name'
        },
        ['key', 'name']
    )

]


@pytest.mark.parametrize('x, match', power_dict_attr_data)
def test_power_dict_attr(x, match):
    test_dict = PowerDict(x)
    for attr in match:
        assert hasattr(test_dict, attr)


power_dict_freeze_melt_data = [
    (
        {
            'key': 'value',
        },
        ['key']
    ),
    (
        {
            'PowerDict': PowerDict({'A': 1}),
        },
        ['PowerDict']
    ),
    (
        {
            'list': [1, 2, 3],
            'tuple': (1, 2, 3)
        },
        ['list', 'tuple']
    ),
    (
        {
            'PowerDict_in_list': [PowerDict({'A': 1}), PowerDict({'B': 2})],
            'PowerDict_in_tuple': (PowerDict({'A': 1}), PowerDict({'B': 2}))
        },
        ['PowerDict_in_list', 'PowerDict_in_tuple']
    )
]


@pytest.mark.parametrize('x, match', power_dict_freeze_melt_data)
def test_power_dict_freeze_melt(x, match):
    test_dict = PowerDict(x)
    test_dict.freeze()
    for attr in match:
        try:
            test_dict[attr] = None
            assert False
        except ValueError:
            pass

    test_dict.melt()
    for attr in match:
        test_dict[attr] = None


power_dict_init_data = [
    {
        'd': None,
        'kwargs': None,
        'match': {}
    },
    {
        'd': None,
        'kwargs': {'new': 'update'},
        'match': {'kwargs': {'new': 'update'}}
    }
]


@pytest.mark.parametrize('test_data', power_dict_init_data)
def test_dict_init(test_data: dict):
    if test_data['kwargs'] is not None:
        assert PowerDict(d=test_data['d'], kwargs=test_data['kwargs']) == test_data['match']
    else:
        assert PowerDict(d=test_data['d']) == test_data['match']


power_dict_set_data = [
    ({'int': 1}, 'int', 2, {'int': 2}),
    ({'list': [1, 2, 3]}, 'list', [4, 5, 6], {'list': [4, 5, 6]}),
    ({'tuple': (1, 2, 3)}, 'tuple', (7, 8, 9), {'tuple': [7, 8, 9]})
]


@pytest.mark.parametrize('x, new_key, new_value, match', power_dict_set_data)
def test_power_dict_set(x, new_key, new_value, match):
    test_dict = PowerDict(x)
    test_dict[new_key] = new_value
    assert test_dict == match


power_dict_set_raises_data = [
    ({'int': 1}, 'int', 2, ValueError, "PowerDict is frozen. 'int' cannot be set."),
    ({'list': [1, 2, 3]}, 'list', [3, 4], ValueError, "PowerDict is frozen. 'list' cannot be set."),
    ({'tuple': (1, 2, 3)}, 'tuple', (3, 4), ValueError, "PowerDict is frozen. 'tuple' cannot be set.")
]


@pytest.mark.parametrize('x, new_key, new_value, error, match', power_dict_set_raises_data)
def test_power_dict_set_raises(x, new_key, new_value, error, match):
    test_dict = PowerDict(x)
    test_dict.freeze()
    with pytest.raises(error, match=match):
        test_dict[new_key] = new_value


power_dict_update_data = [
    ({'a': 1, 'b': 2}, {'b': 4, 'c': 3}, {'a': 1, 'b': 4, 'c': 3}),
    ({'a': 1, 'b': 2}, {'c': [1, 2]}, {'a': 1, 'b': 2, 'c': [1, 2]}),
    ({'a': 1, 'b': 2}, {'c': (1, 2)}, {'a': 1, 'b': 2, 'c': [1, 2]})
]


@pytest.mark.parametrize('x, e, match', power_dict_update_data)
def test_power_dict_update(x, e, match):
    test_dict = PowerDict(x)
    test_dict.update(e)
    assert test_dict == match


power_dict_pop_data = [
    ({'a': 1, 'b': 2, 'c': 3}, 'b', {'a': 1, 'c': 3}),
    ({'a': 1, 'b': 2, 'c': [1, 2]}, 'b', {'a': 1, 'c': [1, 2]}),
    ({'a': 1, 'b': 2, 'c': (1, 2)}, 'b', {'a': 1, 'c': [1, 2]}),
]


@pytest.mark.parametrize('x, key, match', power_dict_pop_data)
def test_power_dict_pop(x, key, match):
    test_dict = PowerDict(x)
    test_dict.pop(key)
    assert test_dict == match


test_power_dict_del_raises = [
    ({'a': 1, 'b': 2, 'c': 3}, 'b', ValueError, "PowerDict is frozen. 'b' cannot be del."),
    ({'a': 1, 'b': 2, 'c': [1, 2]}, 'b', ValueError, "PowerDict is frozen. 'b' cannot be del."),
    ({'a': 1, 'b': 2, 'c': (1, 2)}, 'b', ValueError, "PowerDict is frozen. 'b' cannot be del."),
]


@pytest.mark.parametrize('x, key, error, match', test_power_dict_del_raises)
def test_power_dict_del_raises(x, key, error, match):
    test_dict = PowerDict(x)
    test_dict.freeze()
    with pytest.raises(error, match=match):
        del test_dict[key]
