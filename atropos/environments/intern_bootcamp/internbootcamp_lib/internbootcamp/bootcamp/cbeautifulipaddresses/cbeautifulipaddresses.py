"""# 

### 谜题描述
The problem uses a simplified TCP/IP address model, please read the statement carefully.

An IP address is a 32-bit integer, represented as a group of four decimal 8-bit integers (without leading zeroes), separated by commas. For example, record 0.255.1.123 shows a correct IP address and records 0.256.1.123 and 0.255.1.01 do not. In the given problem an arbitrary group of four 8-bit integers is a correct IP address.

Our hero Polycarpus still works as a system administrator in some large corporation. He likes beautiful IP addresses. To check if some IP address is beautiful, he should do the following:

  1. write out in a line four 8-bit numbers of the IP address, without the commas; 
  2. check if the resulting string is a palindrome. 



Let us remind you that a palindrome is a string that reads the same from right to left and from left to right.

For example, IP addresses 12.102.20.121 and 0.3.14.130 are beautiful (as strings \"1210220121\" and \"0314130\" are palindromes), and IP addresses 1.20.20.1 and 100.4.4.1 are not.

Polycarpus wants to find all beautiful IP addresses that have the given set of digits. Each digit from the set must occur in the IP address at least once. IP address must not contain any other digits. Help him to cope with this difficult task.

Input

The first line contains a single integer n (1 ≤ n ≤ 10) — the number of digits in the set. The second line contains the set of integers a1, a2, ..., an (0 ≤ ai ≤ 9). It is guaranteed that all digits in the set are distinct.

Output

In the first line print a single integer k — the number of beautiful IP addresses that contain the given set of digits. In the following k lines print the IP addresses, one per line in the arbitrary order.

Examples

Input

6
0 1 2 9 8 7


Output

6
78.190.209.187
79.180.208.197
87.190.209.178
89.170.207.198
97.180.208.179
98.170.207.189


Input

1
4


Output

16
4.4.4.4
4.4.4.44
4.4.44.4
4.4.44.44
4.44.4.4
4.44.4.44
4.44.44.4
4.44.44.44
44.4.4.4
44.4.4.44
44.4.44.4
44.4.44.44
44.44.4.4
44.44.4.44
44.44.44.4
44.44.44.44

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys, itertools
sys.setrecursionlimit(10001)

def valid(num):
	n = int(num)
	if str(n) != num:
		return False
	return 0 <= n and n <= 255

def div_ip1(n):
	for x in xrange(2, len(n)-1):
		for ip1 in div_ip2(n[:x]):
			for ip2 in div_ip2(n[x:]):
				yield ip1 + \".\" + ip2

def div_ip2(n):
	for x in xrange(1, len(n)):
		if valid(n[:x]) and valid(n[x:]):
			yield n[:x] + \".\" + n[x:]

def to_ip(n):
	s = \"\"
	for i in n:
		s += str(i)
	return s

def merge(num):
	m.add(to_ip(num))
	if len(num) < 6:
		for a in A:
			for i in xrange(len(num) + 1):
				num.insert(i, a)
				m.add(to_ip(num))
				merge(num)
				num.pop(i)

n = int(sys.stdin.readline().strip())
A = map(int, sys.stdin.readline().strip().split())

ans = set()
if n <= 6:
	for ls in itertools.permutations(A, n):
		num = to_ip(ls)
		m = set()
		merge(list(ls))
		for p in m:
			for x in div_ip1(p + p[::-1]):
				ans.add(x)
			for x in div_ip1(p + p[-2::-1]):
				ans.add(x)
print len(ans)
for ip in ans:
	print ip
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from itertools import permutations
from bootcamp import Basebootcamp

class Cbeautifulipaddressesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=6):
        """
        Initialize bootcamp with parameters for generating puzzle instances.
        :param min_n: Minimum number of distinct digits (1-6)
        :param max_n: Maximum number of distinct digits (1-6)
        """
        self.min_n = max(1, min(min_n, 6))
        self.max_n = min(6, max(max_n, self.min_n))
    
    def case_generator(self):
        """
        Generate valid puzzle instances with guaranteed solutions.
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            n = random.randint(self.min_n, self.max_n)
            digits = random.sample(range(10), n)
            correct_ips = self._calculate_correct_ips(n, digits)
            if correct_ips:
                return {
                    'n': n,
                    'digits': digits
                }
        # Fallback case if no valid case found
        return {
            'n': 1,
            'digits': [4]
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        digits = sorted(question_case['digits'])
        n = question_case['n']
        digits_str = ' '.join(map(str, digits))
        return f"""You are a network security expert analyzing special IP addresses. Find all "beautiful" IP addresses that meet these strict requirements:

1. **Digit Requirements**  
   - Must contain ALL of these digits: {digits_str}  
   - Must NOT contain any other digits

2. **IP Format Validity**  
   - Must be a valid IPv4 address (four 0-255 numbers separated by dots)
   - No leading zeros in any segment (e.g., "012" is invalid)

3. **Palindrome Structure**  
   - When concatenated without dots (e.g., "192.168.1.1" becomes "19216811"), the entire string must read the same backward

**Output Format**  
- First line: Total valid IPs found (k)  
- Next k lines: Each beautiful IP  
- Enclose your answer between [answer] and [/answer]  

Example for digits 0,1,2,7,8,9:
[answer]
6
78.190.209.187
79.180.208.197
87.190.209.178
89.170.207.198
97.180.208.179
98.170.207.189
[/answer]

Current challenge: Find all beautiful IPs containing these {n} digits: {digits_str}"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        try:
            k = int(lines[0])
            ips = lines[1:k+1]
            if len(ips) != k:
                return None
            return ips
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_ips = cls._calculate_correct_ips(identity['n'], identity['digits'])
        return sorted(solution or []) == sorted(correct_ips)

    @classmethod
    def _calculate_correct_ips(cls, n, digits):
        required = set(map(str, digits))
        ips = set()
        
        # Early return for impossible cases
        if n > 6:
            return []
        
        for perm in permutations(digits, n):
            visited = set()
            cls._generate_p_candidates(perm, digits, visited)
            for p in visited:
                # Generate two possible palindrome patterns
                candidates = [
                    p + p[::-1],          # Full mirror (even length)
                    p + p[:-1][::-1]      # Odd length mirror
                ]
                for candidate in candidates:
                    ips.update(cls._split_valid_ips(candidate))
        
        # Validate final IP set
        valid = []
        for ip in ips:
            # Check digit composition
            chars = set(ip.replace('.', ''))
            if chars != required:
                continue
            
            # Check all octets are valid
            if all(cls._valid_octet(octet) for octet in ip.split('.')):
                valid.append(ip)
                
        return sorted(valid)

    @classmethod
    def _generate_p_candidates(cls, num, digits, visited, current=None):
        if current is None:
            current = []
        s = ''.join(map(str, current + list(num)))
        if s in visited:
            return
        visited.add(s)
        if len(s) >= 6:
            return
        for d in digits:
            for i in range(len(current)+1):
                new_current = current[:i] + [d] + current[i:]
                cls._generate_p_candidates(num, digits, visited, new_current)

    @staticmethod
    def _split_valid_ips(s):
        ips = set()
        # Split into 4 octets by dividing the string twice
        for split_pos in range(2, len(s)-1):  # First split position
            left_part = s[:split_pos]
            right_part = s[split_pos:]
            
            # Split left part into 2 octets
            for left_octet1, left_octet2 in Cbeautifulipaddressesbootcamp._split_two(left_part):
                # Split right part into 2 octets
                for right_octet1, right_octet2 in Cbeautifulipaddressesbootcamp._split_two(right_part):
                    ip = f"{left_octet1}.{left_octet2}.{right_octet1}.{right_octet2}"
                    ips.add(ip)
        return ips

    @staticmethod
    def _split_two(s):
        splits = []
        for i in range(1, len(s)):
            a = s[:i]
            b = s[i:]
            if Cbeautifulipaddressesbootcamp._valid_octet(a) and Cbeautifulipaddressesbootcamp._valid_octet(b):
                splits.append((a, b))
        return splits

    @staticmethod
    def _valid_octet(s):
        if not s:
            return False
        if s[0] == '0' and len(s) > 1:  # Leading zero check
            return False
        return 0 <= int(s) <= 255
