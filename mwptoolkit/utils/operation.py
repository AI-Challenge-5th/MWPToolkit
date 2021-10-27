import math
import functools
from typing import List, Any, Tuple, Union
from numbers import Number
from decimal import Decimal
from operator import itemgetter
from itertools import permutations

def type_check(type_a, type_b):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(a, b):
            if isinstance(a, type_a) and isinstance(b, type_b):
                return func(a, b)
            else:
                return None
        return wrapper
    return decorator


def length_check(func):
    @functools.wraps(func)
    def wrapper(a, b):
        if len(a) > b:
            return func(a, b)
        return None
    return wrapper

@type_check(Number, Number)
def add(a: Number, b: Number) -> Number:
    return a + b

@type_check(Number, Number)
def sub(a: Number, b: Number) -> Number:
    return a - b

@type_check(Number, Number)
def mul(a: Number, b: Number) -> Number:
    return a * b

@type_check(Number, Number)
def div(a: Number, b: Number) -> Number:
    return a / b

@type_check(Number, Number)
def gcd(a: Number, b: Number) -> Number:
    if not is_integer(a) or not is_integer(b):
        return None

    return Decimal(math.gcd(int(a), int(b)))

@type_check(Number, Number)
def lcm(a: Number, b: Number) -> Number:
    if not is_integer(a) or not is_integer(b):
        return None
    
    return a * b / Decimal(math.gcd(int(a), int(b)))

@type_check(List, Number)
@length_check
def min_(a: List[Number], b: Number) -> Number:
    if not all(map(lambda x: isinstance(x, Number), a)):
        return None
    
    if not is_integer(b):
        return None
    
    return sorted(a)[int(b)]

@type_check(List, Number)
@length_check
def max_(a: List[Number], b: Number) -> Number:
    if not all(map(lambda x: isinstance(x, Number), a)):
        return None
    
    if not is_integer(b):
        return None
    
    return sorted(a)[::-1][int(b)]

@type_check(List, Number)
@length_check
def argmin(a: List[Tuple[str, Number]], b: Number) -> str:
    if not all(map(lambda x: isinstance(x, Tuple), a)):
        return None
    
    if not is_integer(b):
        return None
    
    return sorted(a, key=itemgetter(1))[int(b)][0]

@type_check(List, Number)
@length_check
def argmax(a: List[Tuple[str, Number]], b: Number) -> str:
    if not all(map(lambda x: isinstance(x, Tuple), a)):
        return None
    
    if not is_integer(b):
        return None
    
    return sorted(a, key=itemgetter(1))[::-1][int(b)][0]

@type_check(List, (Number, List, Tuple, str))
def len_(a: List, b: Any) -> Number:
    return len(a)

@type_check((Number, Tuple, str), (List, Number, Tuple, str))
def concat(a: Union[str, Number, Tuple], b: Union[List, str, Number, Tuple]):
    if isinstance(b, List):
        if all(map(lambda x: isinstance(x, type(a)), b)):
            return [a] + b
        
        return None
        
    if type(a) == type(b):
        return [a, b]
    
    return None

@type_check(str, Number)
def tuple_(a: str, b: Number):
    return (a, b)

def merge(p):
    return sum(map(lambda x: x[1] * 10**x[0], enumerate(p[::-1])))

@type_check(List, Number)
def gen10(a: List[Number], b: Number):
    if not all(map(lambda x: isinstance(x, Number), a)):
        return None
    
    if not is_integer(b):
        return None
    
    permute = [merge(it) for it in permutations(a, int(b)) if it[0] != 0]
    
    return permute

@type_check(Number, Number)
def quo(a: Number, b: Number):
    return a // b

@type_check(Number, Number)
def rem(a: Number, b: Number):
    return a % b    


def is_integer(number):
    return number % 1 == 0





from decimal import Decimal

assert add(Decimal('1'), Decimal('2')) == Decimal('3')
assert add(Decimal('1'), [Decimal('2')]) is None
assert add('1', 2) is None
assert sub(1, 2) == -1
assert sub(1, '2') is None
assert sub('1', 2) is None
assert mul(1, 2) == 2
assert mul(1, '2') is None
assert mul('1', 2) is None
assert div(1, 2) - 1.5 < 1e-6
assert div(1, '2') is None
assert div('1', 2) is None
assert gcd(Decimal(25), Decimal(10.000000000000)) == 5
assert gcd(Decimal(25), Decimal(10.000000000001)) is None
assert lcm(Decimal(3), Decimal(7.000000000000)) == 21
assert lcm(Decimal(3), Decimal(7.000000000001)) is None
assert min_([Decimal(5), Decimal('3.4'), Decimal(14)], 0) == Decimal('3.4')
assert min_([Decimal(5), Decimal('3.4'), Decimal(14)], 1) == Decimal('5')
assert min_([Decimal(5), Decimal('3.4'), Decimal(14)], 2) == Decimal('14')
assert min_([('a', Decimal(4)), Decimal(14)], 2) is None
assert min_(['a', Decimal(4), Decimal(14)], 2) is None
assert max_([Decimal(5), Decimal('3.4'), Decimal(14)], 0) == Decimal('14')
assert max_([Decimal(5), Decimal('3.4'), Decimal(14)], 1) == Decimal('5')
assert max_([Decimal(5), Decimal('3.4'), Decimal(14)], 2) == Decimal('3.4')
assert max_([('a', Decimal(4)), Decimal(14)], 2) is None
assert max_(['a', Decimal(4), Decimal(14)], 2) is None
assert argmin([('지민', Decimal(5)), ('정국', Decimal('3.4')), ('윤아', Decimal(14))], 0) == '정국'
assert argmin([('지민', Decimal(5)), ('정국', Decimal('3.4')), ('윤아', Decimal(14))], 1) == '지민'
assert argmin([('지민', Decimal(5)), ('정국', Decimal('3.4')), ('윤아', Decimal(14))], 2) == '윤아'
assert argmin([('a', Decimal(4)), Decimal(14)], 2) is None
assert argmin(['a', Decimal(4), Decimal(14)], 2) is None
assert argmax([('지민', Decimal(5)), ('정국', Decimal('3.4')), ('윤아', Decimal(14))], 0) == '윤아'
assert argmax([('지민', Decimal(5)), ('정국', Decimal('3.4')), ('윤아', Decimal(14))], 1) == '지민'
assert argmax([('지민', Decimal(5)), ('정국', Decimal('3.4')), ('윤아', Decimal(14))], 2) == '정국'
assert argmax([('a', Decimal(4)), Decimal(14)], 2) is None
assert argmax(['a', Decimal(4), Decimal(14)], 2) is None
assert len_([('지민', Decimal(5)), ('정국', Decimal('3.4')), ('윤아', Decimal(14))], 'a') == 3
assert len_([('a', Decimal(4)), Decimal(14)], 1) == 2
assert len_([Decimal(5), Decimal('3.4'), Decimal(14)], []) == 3
assert concat(1, '정국') is None
assert concat('지민', 2) is None
assert concat('정국', '지민') == ['정국', '지민']
assert concat(123, 456) == [123, 456]
assert concat(('정국', 123), ('지민', 456)) == [('정국', 123), ('지민', 456)]
assert tuple_(1, 2) is None
assert tuple_('정국', '지민') is None
assert tuple_('정국', 23) == ('정국', 23)
assert quo(Decimal('10'), Decimal('3')) == 3
assert rem(Decimal('10'), Decimal('3')) == 1

OPERATIONS = {
    'add': add,
    'sub': sub,
    'mul': mul,
    'div': div,
    'gcd': gcd,
    'lcm': lcm,
    'min': min_,
    'max': max_,
    'argmin': argmin,
    'argmax': argmax,
    'len': len_,
    'concat': concat,
    'tuple': tuple_,
    'gen10': gen10,
    'quo': quo,
    'rem': rem,
}