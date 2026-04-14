"""# 

### 谜题描述
As we all know Barney's job is \"PLEASE\" and he has not much to do at work. That's why he started playing \"cups and key\". In this game there are three identical cups arranged in a line from left to right. Initially key to Barney's heart is under the middle cup.

<image>

Then at one turn Barney swaps the cup in the middle with any of other two cups randomly (he choses each with equal probability), so the chosen cup becomes the middle one. Game lasts n turns and Barney independently choses a cup to swap with the middle one within each turn, and the key always remains in the cup it was at the start.

After n-th turn Barney asks a girl to guess which cup contains the key. The girl points to the middle one but Barney was distracted while making turns and doesn't know if the key is under the middle cup. That's why he asked you to tell him the probability that girl guessed right.

Number n of game turns can be extremely large, that's why Barney did not give it to you. Instead he gave you an array a1, a2, ..., ak such that 

<image>

in other words, n is multiplication of all elements of the given array.

Because of precision difficulties, Barney asked you to tell him the answer as an irreducible fraction. In other words you need to find it as a fraction p / q such that <image>, where <image> is the greatest common divisor. Since p and q can be extremely large, you only need to find the remainders of dividing each of them by 109 + 7.

Please note that we want <image> of p and q to be 1, not <image> of their remainders after dividing by 109 + 7.

Input

The first line of input contains a single integer k (1 ≤ k ≤ 105) — the number of elements in array Barney gave you.

The second line contains k integers a1, a2, ..., ak (1 ≤ ai ≤ 1018) — the elements of the array.

Output

In the only line of output print a single string x / y where x is the remainder of dividing p by 109 + 7 and y is the remainder of dividing q by 109 + 7.

Examples

Input

1
2


Output

1/2


Input

3
1 1 1


Output

0/1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
################# point to notice ###############
####### 2**(a1*a2*a3....)= 2**a1**a2**a3.. ######
####### now you can calculate (2**n)%m ##########
def mod(a,b,m):
    result=1
    # print \"reached\"
    while(b):
        if b&1:
            result=(result*a)%m
        b>>=1
        a=(a*a)%m
	#print b
    # print result
    return result

def solve():
    m=1000000007
    k=input()
    a=map(int,raw_input().split())
    prev=2
    odd=1
    for x in xrange(k):
        if a[x]%2==0 :
            odd=0
            break
    for x in xrange(k):
        prev=mod(prev,a[x],m)
    prev=(prev*(mod(2,m-2,m)))%m
    # print prev
    if odd:
        p=(((prev-1)%m)*(mod(3,m-2,m)))%m
    else:
        p=(((prev+1)%m)*(mod(3,m-2,m)))%m
    q=prev
    print str(p)+\"/\"+str(q)
solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Epleasebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.k_min = params.get('k_min', 1)
        self.k_max = params.get('k_max', 5)
        self.element_min = params.get('element_min', 1)
        self.element_max = params.get('element_max', 10)
        
        # Ensure element range is valid
        if self.element_min > self.element_max:
            raise ValueError("Invalid element range")

    def case_generator(self):
        k = random.randint(self.k_min, self.k_max)
        a = []
        has_even = random.choice([True, False])
        
        # Generate at least one even number if needed
        if has_even:
            even_pos = random.randint(0, k-1)
            # Try generating even number
            even_val = None
            possible_evens = [x for x in range(self.element_min, self.element_max+1) if x % 2 == 0]
            
            if possible_evens:
                even_val = random.choice(possible_evens)
            else:  # Force create even if no available
                even_val = max(self.element_min, 2)
                self.element_max = max(self.element_max, even_val)
            
            for i in range(k):
                if i == even_pos:
                    a.append(even_val)
                else:
                    val = random.randint(self.element_min, self.element_max)
                    # Ensure odd for other positions
                    while val % 2 == 0:
                        val = random.randint(self.element_min, self.element_max)
                    a.append(val)
        else:
            for _ in range(k):
                val = random.randint(self.element_min, self.element_max)
                # Force odd if needed
                if val % 2 == 0:
                    val = val - 1 if val > self.element_min else val + 1
                a.append(max(min(val, self.element_max), self.element_min))

        return {"k": k, "a": a}

    @staticmethod
    def prompt_func(question_case) -> str:
        k = question_case['k']
        a = question_case['a']
        return f"""Calculate the probability after cup swaps. Follow these steps:
1. Compute n = product of array elements
2. Calculate 2^(n-1) modulo {MOD}
3. Apply formula: (2^(n-1) + (-1)^n) / 3
4. Output result as irreducible fraction modulo {MOD}

Input:
{k}
{' '.join(map(str, a))}

Format answer as [answer]x/y[/answer]. Example outputs:
[answer]1/2[/answer]
[answer]0/1[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*/\s*(\d+)\s*\[/answer\]', output)
        return f"{matches[-1][0]}/{matches[-1][1]}" if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            a = identity['a']
            k = identity['k']
            x_sol, y_sol = map(int, solution.split('/'))
            
            # Calculate n as product of array elements (mod φ(MOD) for exponent)
            prev = 2
            for num in a:
                prev = pow(prev, num, MOD)
            
            inv_2 = pow(2, MOD-2, MOD)
            prev = (prev * inv_2) % MOD
            
            has_even = any(num % 2 == 0 for num in a)
            inv3 = pow(3, MOD-2, MOD)
            
            if has_even:
                p = (prev + 1) * inv3 % MOD
            else:
                p = (prev - 1) * inv3 % MOD
            q = prev % MOD
            
            return (x_sol == p) and (y_sol == q)
        except:
            return False
