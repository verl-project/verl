"""# 

### 谜题描述
The problem uses a simplified TCP/IP address model, please make sure you've read the statement attentively.

Polycarpus has found a job, he is a system administrator. One day he came across n IP addresses. Each IP address is a 32 bit number, represented as a group of four 8-bit numbers (without leading zeroes), separated by dots. For example, the record 0.255.1.123 shows a correct IP address and records 0.256.1.123 and 0.255.1.01 do not. In this problem an arbitrary group of four 8-bit numbers is a correct IP address.

Having worked as an administrator for some time, Polycarpus learned that if you know the IP address, you can use the subnet mask to get the address of the network that has this IP addess.

The subnet mask is an IP address that has the following property: if we write this IP address as a 32 bit string, that it is representable as \"11...11000..000\". In other words, the subnet mask first has one or more one bits, and then one or more zero bits (overall there are 32 bits). For example, the IP address 2.0.0.0 is not a correct subnet mask as its 32-bit record looks as 00000010000000000000000000000000.

To get the network address of the IP address, you need to perform the operation of the bitwise \"and\" of the IP address and the subnet mask. For example, if the subnet mask is 255.192.0.0, and the IP address is 192.168.1.2, then the network address equals 192.128.0.0. In the bitwise \"and\" the result has a bit that equals 1 if and only if both operands have corresponding bits equal to one.

Now Polycarpus wants to find all networks to which his IP addresses belong. Unfortunately, Polycarpus lost subnet mask. Fortunately, Polycarpus remembers that his IP addresses belonged to exactly k distinct networks. Help Polycarpus find the subnet mask, such that his IP addresses will belong to exactly k distinct networks. If there are several such subnet masks, find the one whose bit record contains the least number of ones. If such subnet mask do not exist, say so.

Input

The first line contains two integers, n and k (1 ≤ k ≤ n ≤ 105) — the number of IP addresses and networks. The next n lines contain the IP addresses. It is guaranteed that all IP addresses are distinct.

Output

In a single line print the IP address of the subnet mask in the format that is described in the statement, if the required subnet mask exists. Otherwise, print -1.

Examples

Input

5 3
0.0.0.1
0.1.1.2
0.0.2.1
0.1.1.0
0.0.2.3


Output

255.255.254.0

Input

5 2
0.0.0.1
0.1.1.2
0.0.2.1
0.1.1.0
0.0.2.3


Output

255.255.0.0

Input

2 1
255.0.0.1
0.0.0.2


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
    mask_elem = (128, 64, 32, 16, 8, 4, 2, 1)
    cur_mask_block = 0
    ip_list = []
    ip_count, net_count = map(int, raw_input().split())
    for i in xrange(ip_count):
        ip_list.append(map(int, raw_input().split(\".\")))

    for i in xrange(4):
        diff = set()
        for ip in ip_list:
            diff.add(tuple(ip[0:i+1:]))
        if len(diff) >= net_count:
            for j in xrange(8):
                cur_mask_block += mask_elem[j]
                abs_diff = set()
                for ip in ip_list:
                    cip = ip[0:i]
                    cip.append(ip[i] & cur_mask_block)
                    abs_diff.add(tuple(cip))
                l = len(abs_diff)
                if l == net_count:
                    mask = \"255.\" * i
                    mask += str(cur_mask_block) + \".\"
                    mask += \"0.\" * (4 - i - 1)
                    print mask.strip(\".\")
                    return
                if l > net_count:
                    print -1
                    return

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def compute_mask(ip_list, net_count):
    mask_elem = (128, 64, 32, 16, 8, 4, 2, 1)
    for i in range(4):
        diff = set()
        for ip in ip_list:
            diff.add(tuple(ip[:i+1]))
        if len(diff) >= net_count:
            cur_mask_block = 0
            for j in range(8):
                cur_mask_block += mask_elem[j]
                abs_diff = set()
                for ip in ip_list:
                    cip = list(ip[:i])
                    current_octet = ip[i] & cur_mask_block
                    cip.append(current_octet)
                    abs_diff.add(tuple(cip))
                current_network_count = len(abs_diff)
                if current_network_count == net_count:
                    mask_parts = ['255'] * i
                    mask_parts.append(str(cur_mask_block))
                    mask_parts.extend(['0'] * (3 - i))
                    return '.'.join(mask_parts)
                elif current_network_count > net_count:
                    return '-1'
            return '-1'
    return '-1'

class Cnetworkmaskbootcamp(Basebootcamp):
    def __init__(self, ip_count=5, net_count=3, max_octet=255, allow_unsolvable=True):
        self.ip_count = ip_count
        self.net_count = net_count
        self.max_octet = max_octet
        self.allow_unsolvable = allow_unsolvable
    
    def case_generator(self):
        if self.allow_unsolvable and random.random() < 0.3:
            return self._gen_unsolvable_case()
        else:
            return self._gen_solvable_case()
    
    def _gen_unsolvable_case(self):
        ips = [
            [255, random.randint(0,255), random.randint(0,255), random.randint(0,255)],
            [0,   random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        ]
        while ips[0] == ips[1]:
            ips[1] = [0, random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        return {
            'n': 2,
            'k': 1,
            'ips': ips
        }
    
    def _gen_solvable_case(self):
        mask_m = random.randint(8, 28)
        network_prefixes = set()
        while len(network_prefixes) < self.net_count:
            prefix = random.getrandbits(mask_m)
            network_prefixes.add(prefix)
        
        ips = []
        ips_set = set()
        
        # Generate unique IPs for network prefixes
        for prefix in network_prefixes:
            while True:
                new_ip = self.generate_ip(prefix, mask_m)
                ip_tuple = tuple(new_ip)
                if ip_tuple not in ips_set:
                    ips.append(new_ip)
                    ips_set.add(ip_tuple)
                    break
        
        # Generate remaining IPs
        remaining = self.ip_count - self.net_count
        while remaining > 0:
            prefix = random.choice(list(network_prefixes))
            new_ip = self.generate_ip(prefix, mask_m)
            ip_tuple = tuple(new_ip)
            if ip_tuple not in ips_set:
                ips.append(new_ip)
                ips_set.add(ip_tuple)
                remaining -= 1
        
        random.shuffle(ips)
        return {
            'n': self.ip_count,
            'k': self.net_count,
            'ips': ips
        }
    
    def generate_ip(self, prefix, mask_m):
        prefix_bits = bin(prefix)[2:].zfill(mask_m)
        remaining_bits = ''.join(random.choices(['0', '1'], k=32 - mask_m))
        full_bits = prefix_bits + remaining_bits
        octets = [int(full_bits[i:i+8], 2) for i in range(0, 32, 8)]
        return octets
    
    @staticmethod
    def prompt_func(question_case):
        ips = question_case['ips']
        k = question_case['k']
        ips_str = "\n".join(".".join(map(str, ip)) for ip in ips)
        prompt = f"""You are a system administrator. You have {len(ips)} distinct IP addresses. Your task is to find a valid subnet mask that groups these IPs into exactly {k} distinct networks. The subnet mask must be a valid IP address where the binary representation consists of consecutive 1's followed by consecutive 0's. If multiple masks are possible, choose the one with the fewest 1's. If impossible, output -1.

The IP addresses are:
{ips_str}

Format your answer as [answer]<subnet_mask_or_-1>[/answer]. Example:
[answer]255.255.254.0[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match == '-1':
            return '-1'
        try:
            parts = list(map(int, last_match.split('.')))
            if len(parts) != 4:
                return None
            for p in parts:
                if p < 0 or p > 255:
                    return None
            binary_str = ''.join(f'{p:08b}' for p in parts)
            if not re.match(r'^1+0+$', binary_str):
                return None
            return last_match
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        ips = identity['ips']
        k = identity['k']
        expected = compute_mask(ips, k)
        return solution == expected
