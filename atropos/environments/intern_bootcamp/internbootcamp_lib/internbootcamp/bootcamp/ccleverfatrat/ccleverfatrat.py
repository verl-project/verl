"""# 

### 谜题描述
The Fat Rat and his friend Сerealguy have had a bet whether at least a few oats are going to descend to them by some clever construction. The figure below shows the clever construction.

<image>

A more formal description of the clever construction is as follows. The clever construction consists of n rows with scales. The first row has n scales, the second row has (n - 1) scales, the i-th row has (n - i + 1) scales, the last row has exactly one scale. Let's number the scales in each row from the left to the right, starting from 1. Then the value of wi, k in kilograms (1 ≤ i ≤ n; 1 ≤ k ≤ n - i + 1) is the weight capacity parameter of the k-th scale in the i-th row. 

If a body whose mass is not less than wi, k falls on the scale with weight capacity wi, k, then the scale breaks. At that anything that the scale has on it, either falls one level down to the left (if possible) or one level down to the right (if possible). In other words, if the scale wi, k (i < n) breaks, then there are at most two possible variants in which the contents of the scale's pan can fall out: all contents of scale wi, k falls either on scale wi + 1, k - 1 (if it exists), or on scale wi + 1, k (if it exists). If scale wn, 1 breaks, then all its contents falls right in the Fat Rat's claws. Please note that the scales that are the first and the last in a row, have only one variant of dropping the contents.

Initially, oats are simultaneously put on all scales of the first level. The i-th scale has ai kilograms of oats put on it. After that the scales start breaking and the oats start falling down in some way. You can consider everything to happen instantly. That is, the scale breaks instantly and the oats also fall instantly.

The Fat Rat is sure that whatever happens, he will not get the oats from the first level. Cerealguy is sure that there is such a scenario, when the rat gets at least some number of the oats. Help the Fat Rat and the Cerealguy. Determine, which one is right.

Input

The first line contains a single integer n (1 ≤ n ≤ 50) — the number of rows with scales.

The next line contains n space-separated integers ai (1 ≤ ai ≤ 106) — the masses of the oats in kilograms.

The next n lines contain descriptions of the scales: the i-th line contains (n - i + 1) space-separated integers wi, k (1 ≤ wi, k ≤ 106) — the weight capacity parameters for the scales that stand on the i-th row, in kilograms.

Output

Print \"Fat Rat\" if the Fat Rat is right, otherwise print \"Cerealguy\".

Examples

Input

1
1
2


Output

Fat Rat


Input

2
2 2
1 2
4


Output

Cerealguy


Input

2
2 2
1 2
5


Output

Fat Rat

Note

Notes to the examples: 

  * The first example: the scale with weight capacity 2 gets 1. That means that the lower scale don't break. 
  * The second sample: all scales in the top row obviously break. Then the oats fall on the lower row. Their total mass is 4,and that's exactly the weight that the lower scale can \"nearly endure\". So, as 4  ≥  4, the scale breaks.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#     Clever Fat Rat
import time
from datetime import datetime

import itertools

max_oats = 10**6 + 1

def is_goal(state):
    (a, ws) = state
    if(len(ws) == 1 and ws[0][0] <= a[0]):
        return True
        
def child(state):
    (a, ws) = state 
    scales = ws[0]
    if(len(ws) == 1):
        return [];
    
    b = [[0]*(len(scales)-1)]
    for (jdx, scale) in enumerate(scales):
        if a[jdx] >= scale:
            if(jdx == 0):
                b = map(lambda x: [x[0]+a[jdx]] + x[1:], b)
#                b[0] += a[jdx]
            elif (jdx == len(scales)-1):
                b = map(lambda x:  x[:-1]+[x[-1]+a[-1]], b)
#                b[-1] += a[-1]
            else:
                b = map(lambda x:  [x[:jdx]+[x[jdx]+a[jdx]]+x[jdx+1:], x[:jdx-1]+[x[jdx-1]+a[jdx]]+x[jdx:]], b)
                b = reduce (lambda a,b: a+b, b)
#                b[jdx-1] += a[jdx]/2.
#                b[jdx] += a[jdx]/2.
#    print b
    return map(lambda x: (x, ws[1:]), b)
    
def is_possible_reach_goal(state):
#    print state
    (a, ws) = state
    return (sum(a) >= ws[-1][0])

def is_broken(i, a, ws):
    return 1 if (a[i] >= ws[0][i]) else 0

def fall(idx, a, ws):
    return a[idx]  * is_broken(idx, a, ws)

def check_break(a, ws):
    break_list = [[0,0]]
    for idx in range(0, len(a)-1):
        break_list += [[0,0]]
        if(fall(idx, a, ws) + fall(idx+1, a, ws) >= ws[1][idx]):
            break_list[idx][1] = is_broken(idx, a, ws)
            break_list[idx+1][0] = is_broken(idx+1, a, ws)
#    print break_list
    return break_list

def next_step(a, break_list):
    next_step = [[]]
    for idx, b in enumerate(break_list):
        if b == [0,0]:
            next_step = map((lambda x: x+[0]), next_step)
        elif b == [0,1]:
            next_step = map((lambda x: x+[a[idx]]), next_step)
        elif b == [1,0]:
            next_step = map((lambda x: x[:-1] + [x[-1]+a[idx]] +[0]), next_step)
        else:   # [1,1]
            next_step = map((lambda x: [x[:-1] + [x[-1]+a[idx]] +[0], x+[a[idx]]]), next_step)
            next_step = reduce (lambda a,b: a+b, next_step)
    return map(lambda x: x[:-1],next_step)
                
def weight(p):
    return p[0]

def right(p):
    return p[1][1]

def left(p):
    return p[1][0]

def plus(l, r):
    return (weight(l)+weight(r), (left(l), right(r)))

# 01 can cover 02?
def cover(o1, o2):
    return left(o1) >= left(o2) and right(o1) <= right(o2) and weight(o1) >= weight(o2)

def add_only_unique(oats, oat):
    for idx, o_n in enumerate(oats):
        if oat == o_n or cover(o_n, oat):
            return oats
        if cover(oat, o_n):
            oats[idx] = oat
            return oats
    oats.append(oat)
    return oats

def unique(oats):
    new_oats = []
    for o in oats:
#        if o not in new_oats:
        should_add = True
        for idx, o_n in enumerate(new_oats):
            if o == o_n or cover(o_n, o):
                should_add = False 
                break
            if cover(o, o_n):
                should_add = False 
                new_oats[idx] = o
        if(should_add):
            new_oats.append(o)
    return new_oats
            
def max_drop(oats):
    max_drop = 0
    for oat in oats:
        max_drop = weight(oat) if  weight(oat) > max_drop else max_drop
    return max_drop

def print_now(message):
    print message,
    print datetime.now().strftime(u'%H:%M:%S:%f')

def possible_oats(a, ws):
#    print_now(\"> possible_oats\") 
    oats_list = []
    for idx in range(0, len(a)-1):  # same as len(ws)
#        print_now(\"> possible_oats loop\") 
        oats = []
        left_oat = a[idx]
        right_only_oat = [e for e in a[idx+1] if not e in left_oat ]
        left_only_oat = [e for e in a[idx] if not e in a[idx+1] ]
#        print len(right_only_oat)
#        print len(left_only_oat)
        
#        max_r = max_drop(a[idx+1])
        if ((len(a[idx+1]) == 0) or
            (idx != len(a)-2 and max_drop(a[idx+1])+max_drop(a[idx+2]) >= ws[0][idx+1])):
            for l in left_oat:
                if weight(l) >= ws[0][idx]:
                    oats.append(l)
                        
        if ((len(a[idx]) == 0) or
            (idx != 0 and len(oats_list[-1]) != 0)):
            for r in right_only_oat:
                if weight(r) >= ws[0][idx]:
                    oats = add_only_unique(oats, r)
#        print_now(\"> possible_oats - pre for\") 
#        print len(a),
#        print len(oats),
#        print len(left_only_oat),
#        print len(right_only_oat)
        
        for c in itertools.product(left_only_oat,right_only_oat):
            (l,r) = c
            if weight(r)+weight(l) >= ws[0][idx] and right(l) < left(r):
                oats = add_only_unique(oats, plus(l,r))
#        print_now(\"> possible_oats - unique\") 
        if len(oats) > 30:
            oats.sort(key=weight, reverse=True) 
            oats = oats[:30]
#            print oats
        oats_list.append(oats)   
#    print_now(\"> possible_oats - return\") 
    return oats_list
                
def is_break_all(limit, oats_list):
    for idx, oat in enumerate(oats_list):
        for o in oat:
            if weight(o) >= limit[idx]:
                return True
    return False
    
def fatrat(state):
#    print state

    (a, ws) = state
#    if len(a) == 11:
#        return \"Fat Rat\"

    
    ws = map(lambda x: [max_oats]+x+[max_oats], ws)

    goal_oats = []
    pre_goal_oat = [0, 0]
    for idx in range(len(ws)-1,-1,-1):
        goal_oat = []
        for jdx in range(1,len(ws[idx])-1):
            goal_oat.append(max(ws[idx][jdx],min(pre_goal_oat[jdx-1], pre_goal_oat[jdx])))
        goal_oats.append(goal_oat)
        pre_goal_oat = [max_oats] + goal_oat + [max_oats]
    goal_oats.reverse()
    ws = map(lambda x: x[1:-1], ws)
#    print goal_oats

    oats_list = []
    for idx in range(0, len(a)):
        if a[idx] >= ws[0][idx]:
            oats_list.append([(a[idx], (idx,idx))])
        else:
            oats_list.append([])
    
    repeat = 0
    while(True):
        ws = ws[1:]
        if not len(ws):
            if len(oats_list[0]):
                return \"Cerealguy\"
            else:
                return \"Fat Rat\"
            
#        print goal_oats[0]
#        print oats_list
        if is_break_all(goal_oats[0], oats_list):
            return \"Cerealguy\"
            
        oats_list = possible_oats(oats_list, ws)
        goal_oats = goal_oats[1:]

#        repeat +=1
#        if repeat > 20:
#            print oats_list
#            return \"Finish\"
                           
                             
        
    
    
    
def fatrat3(state):
    print state
    (a, ws) = state
    
    a_list = [a]
    while(True):
        print len(ws)
        print len(a_list)
        print
#        print a_list[0:20]
#        if len(a_list) > 100:
#            break
        if len(ws) == 1:
            for e_a_list in a_list:
                if e_a_list[0] >= ws[0][0]:
                    return \"Cerealguy\"
            return \"Fat Rat\"
        
        a_list = map((lambda a: next_step(a, check_break(a, ws))), a_list)
        a_list = reduce (lambda a,b: a+b, a_list)
        
        new_list = []
        for e_a_list in a_list:
            if e_a_list not in new_list:
                if sum(e_a_list) >= ws[-1][0]:
                    new_list.append(e_a_list)
                
        a_list = new_list
        ws = ws[1:]
        
        if not len(a_list):
            return \"Fat Rat\"
            
def create_goals(ws):
    ws = map(lambda x: [max_oats]+x+[max_oats], ws)

    goal_oats = []
    pre_goal_oat = [0, 0]
    for idx in range(len(ws)-1,-1,-1):
        goal_oat = []
        for jdx in range(1,len(ws[idx])-1):
            goal_oat.append(max(ws[idx][jdx],min(pre_goal_oat[jdx-1], pre_goal_oat[jdx])))
        goal_oats.append(goal_oat)
        pre_goal_oat = [max_oats] + goal_oat + [max_oats]
    goal_oats.reverse()
#    ws = map(lambda x: x[1:-1], ws)
    return goal_oats
#    print goal_oats

def compare_goal(goals, oats):
    for idx in range(0,len(goals)):
        if(goals[idx] <= oats[idx]):
            return True
    return False
    
def fatrat2(state):
    stack = [state]
    visited = []
    start = datetime.now()
    
    goals = create_goals(state[1])
    while(stack):

        state = stack.pop(0)
        visited += [state[0]]
        
#        if(is_goal(state)):
        if(compare_goal(goals[len(goals)-len(state[0])], state[0])):
            return \"Cerealguy\"
        
        children = child(state)
        for c in children:
            if(c[0] not in visited and c not in stack):
                if is_possible_reach_goal(c):
                    stack += [c]
   
    return \"Fat Rat\" 

Debug = False
if(not Debug):
    n = int(raw_input())
    a_s = raw_input().split()
    a = []
    for a_s_e in a_s:
        a += [int(a_s_e)]
    
    ws = []
    for i in range(0, int(n)):
        r = raw_input().split()
#        rs = [max_oats]
        rs = []
        for e_rs in r:
            rs += [int(e_rs)]
#        rs += max_oats
        ws += [rs]
#    print fatrat(n, a, ws)
    print fatrat(((a, ws)))
else:
    worst_w = [[50]]
    for idx in range(1,50):
        worst_w.append([1]*(idx+1))
    worst_w.reverse()
#    print fatrat(([1]*50, worst_w))   #F
 
    print fatrat(([1,1,1,1,1,1,1],[[1,9,1,9,1,9,9],[1,9,1,9,1,9],[1,1,9,1,9],[1,9,9,1],[1,9,1],[1,1],[3]])) #C
    print fatrat(([1, 1, 2, 2, 1, 1], [[1, 1, 2, 2, 1, 1],[2, 4, 2, 4, 2],[2,2,2,2],[4,10,4],[4,4],[8]]))   #F
    print fatrat(([1, 1, 1], [[1, 1, 1],[1, 1],[4]]))   #F
    print fatrat(([2, 2, 2], [[2, 2, 2],[3, 2],[4]]))   #C
    print fatrat(([1], [[1]]))   #C
    print fatrat(([2, 2], [[1, 2],[4]]))   #C
    print fatrat(([2, 2], [[1, 2],[5]]))   #F
    print fatrat(([798097, 886901, 292688, 792934], [[987579, 447910, 689959, 311317],[41624, 797440, 706737],[921438, 988902],[506461]]))   #C
    a =( [232602, 103849, 827367, 389557, 651438, 216320, 824798, 525699, 23338, 518302, 719391, 553814, 331160, 617684, 289434, 312706, 618709, 259095, 21269, 998945, 461731]
    + [896427, 149781, 499724, 6493, 239022, 333269, 513014, 671173, 502655, 287667, 863972, 857850, 644809, 824618, 402957, 617413, 295280, 915642, 78666, 498130, 693142])
    #0
    w =[0]*42
    w[0]=( [442788, 122361, 376827, 1098, 885713, 765876, 370112, 54990, 458771, 438057, 765395, 895171, 272899, 408086, 963600, 961459, 805320, 99236, 780298, 932795, 511481]
     +[195036, 855105, 514771, 711514, 234442, 539631, 644411, 463491, 112557, 217985, 629316, 185503, 888215, 728675, 175993, 704847, 245992, 469172, 819496, 608942, 786465])
#    W1=( [604957, 444979, 92612, 722708, 474069, 634935, 49008, 727286, 15642, 757260, 163229, 242680, 662984, 151936, 302866, 970105, 42818, 86986, 542819, 152685])
    #+ [614993, 744625, 774390, 147357, 217239, 448556, 977399, 440373, 650208, 929115, 60946, 434417, 20...
#    w1 = ([806141, 604957, 444979, 92612, 722708, 474069, 634935, 49008, 727286, 15642, 757260, 163229, 242680, 662984, 151936, 302866, 970105, 42818, 86986, 542819, 152685]
#[765593, 147600, 186480, 720359, 234733, 364648, 8995, 884055, 565526, 558538, 319000, 388...
    w[1] = ([765593, 147600, 186480, 720359, 234733, 364648, 8995, 884055, 565526, 558538, 319000, 388544, 274611, 872762, 244955, 981313, 877169, 440983, 367837, 367936]
    + [806141, 604957, 444979, 92612, 722708, 474069, 634935, 49008, 727286, 15642, 757260, 163229, 242680, 662984, 151936, 302866, 970105, 42818, 86986, 542819, 152685])

    w[2]=( [103072, 936498, 847907, 650645, 566772, 244240, 76487, 607887, 833591, 261100, 535448, 472137, 921365, 782021, 699092, 571729, 884498, 898861, 570530, 8136, 278423, 614993, 744625, 774390, 147357, 217239, 448556, 977399, 440373, 650208]
    + [929115, 60946, 434417, 203564, 373657, 245610, 284531, 327005, 518126, 979469])
    
    w[3]= ([415376, 150228, 72136, 403305, 640672, 652152, 214427, 737311, 208036, 769173, 693842, 421040, 183828, 647456, 73520, 674069, 253765, 239577, 992072, 247531, 5556, 775497, 835157, 659594, 777970, 399659, 357111, 242550, 765227, 396071]
    + [337931, 684782, 912212, 59641, 407013, 892962, 529009, 168624, 729261])
    
    w[4]=([579778, 603392, 7187, 711763, 980786, 891205, 187614, 347504, 871321, 16499, 165802, 430266, 767897, 943796, 838570, 489956, 126553, 519253, 411089, 156752, 209661, 853585, 233490, 370034, 817847, 507577, 999103, 22978, 790519, 52028]
    +[728211, 18397, 740606, 627417, 513419, 851193, 795920, 975646])
    
    w[5]=([364270, 878531, 313352, 760219, 57774, 979287, 495749, 821992, 400807, 118875, 624706, 11664, 398468, 399161, 480516, 516761, 361857, 676791, 254124, 676721, 383882, 346228, 368172, 543223, 372419, 89729, 450007, 183906, 578337, 782425]
    +[239888, 133714, 848309, 211198, 276004, 422001, 342793])
    
    w[6]=(
    [669810, 70004, 223430, 545763, 495696, 659851, 458070, 320934, 70213, 2510, 670903, 398360, 955053, 229649, 10354, 681727, 93774, 345990, 810132, 724068, 178347, 919760, 900818, 664569, 731142, 68854, 562838, 348118, 412534, 711858]
    +[217450, 390687, 476960, 552361, 384448, 269506])
    
    w[7]= ([686869, 943862, 411867, 608029, 563528, 355743, 650398, 806908, 984874, 442601, 965871, 224704, 395748, 333576, 560520, 44827, 222000, 210286, 63258, 240375, 156949, 653188, 279453, 955766, 955005, 475318, 105273, 315988, 259444, 554025]
    +[169162, 673513, 337343, 217480, 802836])
    
    w[8]=([524812, 846231, 854588, 192451, 593937, 58689, 804666, 287964, 117042, 310411, 495542, 750086, 835403, 781732, 699317, 778671, 275391, 903769, 90240, 110812, 176953, 320104, 584756, 341201, 623117, 929603, 391734, 803929, 440316, 17762]
    +[349381, 33395, 717334, 254917])
    
    w[9]=([852392, 358681, 885946, 1989, 714822, 291374, 275129, 793814, 698964, 135683, 654809, 411164, 475752, 828614, 173173, 465830, 881709, 429319, 942530, 613387, 394352, 150504, 429484, 432671, 47638, 720929, 915486, 178642, 362160, 697774]
    +[582177, 990694, 63463])
    
    w[10]=([775829, 967068, 76364, 48338, 231255, 628921, 800537, 721127, 331827, 449577, 987306, 400525, 752721, 529533, 823633, 106767, 436626, 205685, 547258, 868397, 522545, 178774, 793439, 158477, 194198, 537780, 714979, 545295, 240759, 136464]
    +[342543, 669949])
    
    w[11]=([395243, 682010, 637544, 537377, 568397, 939159, 617065, 455601, 589065, 520285, 39074, 59942, 70214, 98748, 330093, 61675, 314160, 4343, 820744, 569072, 156641, 964848, 619284, 939421, 248299, 973729, 72425, 651620, 337476, 228294]
    +[446228])
    w[12] = [733423, 120527, 880793, 28654, 481273, 936515, 808693, 646362, 891193, 526586, 645212, 971930, 320632, 217240, 232724, 519503, 911492, 119638, 207340, 834315, 393853, 595370, 182300, 154260, 490060, 679573, 469578, 931192, 795252, 794504]
    w[13] = [992937, 935048, 234752, 790452, 581692, 261170, 364764, 938989, 760569, 392383, 343936, 652090, 226824, 806565, 456839, 687163, 528042, 195594, 873694, 66359, 594999, 965206, 59095, 345665, 918463, 273203, 35037, 899079, 842258]
    w[14]=[529636, 112898, 877687, 806044, 60735, 189749, 160846, 725032, 512840, 609679, 511, 993861, 575660, 560076, 254140, 79046, 330071, 166742, 370339, 396898, 344306, 315773, 868616, 819708, 981564, 686084, 252763, 573950]
    w[15]=[928772, 710357, 920822, 465694, 16512, 799572, 971974, 162499, 157133, 294992, 241747, 325056, 798852, 528918, 760749, 755326, 867061, 621152, 114485, 917241, 244803, 775595, 547781, 427681, 292956, 933798, 764221]
    w[16]=[406560, 631571, 780422, 992747, 904507, 669339, 120822, 171670, 667218, 377506, 620268, 569831, 150401, 606727, 427415, 149877, 467369, 558074, 701300, 941164, 905161, 621807, 105600, 109208, 227333, 278783]
    w[17]=[844232, 600660, 956873, 275867, 943068, 825161, 895623, 436781, 797538, 72677, 546140, 158861, 475846, 930202, 119313, 70300, 894159, 977232, 72700, 731713, 575949, 152873, 737159, 830782, 65071]
    w[18]=[237486, 549512, 427886, 634225, 662555, 1168, 679325, 898994, 922755, 757632, 664644, 637155, 527137, 507924, 980575, 388955, 210941, 125059, 22317, 78152, 38200, 739583, 554051, 627713]
    w[19]=[479028, 892887, 24398, 430711, 392354, 172860, 835107, 14025, 685353, 867105, 40260, 188534, 475120, 432763, 315378, 632428, 97800, 678056, 637610, 606208, 221689, 756844, 982589]
    w[20]=[496311, 545320, 850902, 922063, 260033, 790113, 586049, 527404, 349253, 221895, 828342, 465666, 633175, 282389, 519791, 331199, 728647, 567952, 2615, 162184, 714835, 235531]
    w[21]=[186477, 32274, 159996, 18134, 428838, 997537, 290934, 736388, 681330, 933659, 158840, 221272, 113015, 443400, 254260, 225996, 711610, 257129, 157857, 458557, 87869]
    w[22]=[601009, 200243, 37750, 427458, 155172, 379052, 313175, 219565, 161355, 632012, 541089, 436743, 196945, 780260, 562593, 479, 977598, 261257, 459383, 188043]
    w[23]=[397, 325219, 455107, 109068, 14827, 610661, 275371, 915727, 386383, 761111, 574986, 371344, 639217, 380366, 910264, 143066, 89832, 111998, 900082]
    w[24]=[54569, 953315, 43302, 252838, 348912, 495555, 823785, 117598, 949, 228308, 699837, 421190, 302746, 843111, 189564, 330203, 411320, 382797]
    w[25]=[285720, 115251, 233252, 448050, 374380, 478883, 644371, 21203, 879789, 5904, 534234, 486941, 73215, 738517, 214733, 114020, 753636]
    w[26]=[674062, 749897, 823484, 423759, 75711, 302798, 112824, 642967, 508005, 530628, 866877, 950023, 649878, 761438, 473985, 474066]
    w[27]=[764631, 542492, 923110, 543523, 771130, 380714, 749869, 488147, 391730, 58854, 709912, 984884, 156149, 574527, 865313]
    w[28]=[285322, 581452, 403449, 558544, 440730, 791866, 963533, 797693, 557146, 188615, 805740, 869602, 58823, 974248]
    w[29]=[541348, 328412, 178390, 827724, 8540, 807870, 861826, 343941, 825017, 837266, 74180, 685457, 162275]
    w[30]=[746285, 307489, 744197, 268642, 453102, 888605, 112405, 860770, 926547, 858125, 451650, 520853]
    w[31]=[750071, 509514, 692960, 418551, 57836, 476673, 681027, 636141, 1841, 259624, 120838]
    w[32]=[260295, 857040, 643083, 533047, 108837, 61847, 849988, 290, 137973, 706032]
    w[33]=[54921, 310584, 531482, 398431, 714080, 21340, 948456, 37528, 652013]
    w[34]=[913611, 142924, 415931, 772076, 351461, 16265, 459565, 738745]
    w[35]=[890204, 456669, 440916, 421772, 768120, 173826, 433774]
    w[36]=[358083, 892804, 903312, 679503, 168920, 921433]
    w[37]=[317439, 489653, 710322, 371652, 567242]
    w[38]=[75849, 947287, 694039, 831792]
    w[39]=[759605, 295166, 463625]
    w[40]=[539029, 755676]
    w[41]=[954370]

#    print fatrat((a, w))   #C
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random
from functools import reduce

max_oats = 10**6 + 1

def create_goals(ws):
    wrapped_ws = []
    for row in ws:
        new_row = [max_oats] + row + [max_oats]
        wrapped_ws.append(new_row)
    goal_oats = []
    pre_goal_oat = [0, 0]
    for idx in range(len(wrapped_ws)-1, -1, -1):
        goal_oat = []
        for jdx in range(1, len(wrapped_ws[idx])-1):
            current_ws = wrapped_ws[idx][jdx]
            left_parent = pre_goal_oat[jdx-1]
            right_parent = pre_goal_oat[jdx]
            goal_value = max(current_ws, min(left_parent, right_parent))
            goal_oat.append(goal_value)
        goal_oats.append(goal_oat)
        pre_goal_oat = [max_oats] + goal_oat + [max_oats]
    goal_oats.reverse()
    return goal_oats

def possible_oats(oats_list, current_ws):
    new_oats = []
    for idx in range(len(current_ws)):
        current_threshold = current_ws[idx]
        available_mass = sum([m for (m, _) in oats_list[idx]])
        if available_mass >= current_threshold:
            left = oats_list[idx-1] if idx > 0 else None
            right = oats_list[idx] if idx < len(oats_list)-1 else None
            new_mass = available_mass
            if left is not None:
                new_left = left + [(new_mass, (idx-1, idx))]
                new_oats.append(new_left)
            if right is not None:
                new_right = right + [(new_mass, (idx, idx+1))]
                new_oats.append(new_right)
    return new_oats

def is_break_all(goal_layer, oats_list):
    for idx, threshold in enumerate(goal_layer):
        if idx >= len(oats_list):
            continue
        total_mass = sum([m for (m, _) in oats_list[idx]])
        if total_mass >= threshold:
            return True
    return False

def fatrat(state):
    try:
        a, ws = state['a'], state['ws']
        goals = create_goals(ws)
        current_layer = [[(m, (0, i))] for i, m in enumerate(a)]
        
        for level in range(len(ws)):
            current_goal = goals[level]
            if is_break_all(current_goal, current_layer):
                return "Cerealguy"
            if level == len(ws)-1:
                break
            current_layer = possible_oats(current_layer, ws[level])
            if not current_layer:
                break
        
        final_check = any(len(grp) > 0 for grp in current_layer)
        return "Cerealguy" if final_check else "Fat Rat"
    except:
        return "Fat Rat"

class Ccleverfatratbootcamp(Basebootcamp):
    def __init__(self, max_n=50, min_weight=1, max_weight=10**6):
        self.max_n = max_n
        self.min_weight = min_weight
        self.max_weight = max_weight

    def case_generator(self):
        n = random.randint(1, self.max_n)
        a = [random.randint(self.min_weight, self.max_weight) for _ in range(n)]
        ws = []
        for i in range(n):
            row_length = n - i
            row = [random.randint(self.min_weight, self.max_weight) for _ in range(row_length)]
            ws.append(row)
        return {'n': n, 'a': a, 'ws': ws}

    @staticmethod
    def prompt_func(question_case) -> str:
        desc = (
            "The Fat Rat and Cerealguy's Scale Puzzle\n\n"
            "Structure Rules:\n"
            "1. There are n rows of scales forming a pyramid\n"
            "2. Each scale breaks if oat mass ≥ its capacity\n"
            "3. Broken scales distribute oats to lower scales\n"
            "4. Final result shows if any oats reach the bottom\n\n"
            f"Input:\n- Row count: {question_case['n']}\n"
            f"- Top row oats: {' '.join(map(str, question_case['a']))}\n"
            "Scale capacities:\n"
        )
        for i, row in enumerate(question_case['ws'], 1):
            desc += f"Row {i}: {' '.join(map(str, row))}\n"
        desc += (
            "\nOutput format: Put your answer (either 'Fat Rat' or 'Cerealguy') "
            "between [answer] and [/answer] tags."
        )
        return desc

    @staticmethod
    def extract_output(output):
        matches = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            output, 
            re.DOTALL
        )
        if not matches:
            return None
        answer = matches[-1].strip()
        if answer.upper() == 'FAT RAT':
            return 'Fat Rat'
        elif answer.upper() == 'CEREALGUY':
            return 'Cerealguy'
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            correct = fatrat(identity)
            return solution == correct
        except:
            return False
